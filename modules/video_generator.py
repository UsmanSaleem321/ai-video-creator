"""Image-to-video generation module.

Backend: Stable Video Diffusion (SVD / SVD-XT via diffusers). A future
phase may add an AnimateDiff-based backend for prompt-driven generation
beyond what single-image SVD conditioning can express.

Design for future extension:
  - `VideoGenerator` is the public-facing class main.py calls; it currently
    hard-wires the SVD backend, but its constructor accepts a `backend`
    argument (only "svd" is implemented; anything else raises
    NotImplementedError) so a later phase can add "animatediff" without
    changing the class's public method signatures.
  - Motion control: `generate_clip_for_scene` accepts an optional
    `motion_hint: Optional[dict] = None` (pre-generation SVD
    motion_bucket_id/noise_aug_strength — intensity only, see
    modules/motion_controller.py), plus optional `camera_motion_type` /
    `action_plan` (post-generation frame-space camera transforms, applied
    via modules.motion_controller.MotionController after SVD renders the
    base clip — this is where actual *directional* motion control comes
    from, since SVD itself has none).
"""

from pathlib import Path
from typing import List, Optional

from PIL import Image
from tqdm.auto import tqdm

from config import Config
from utils.helpers import clear_gpu_memory, timer, retry

try:
    import torch
    from diffusers import StableVideoDiffusionPipeline
except ImportError:
    torch = None
    StableVideoDiffusionPipeline = None


class VideoGenerationError(RuntimeError):
    pass


class VideoGenerator:
    """Turns a single keyframe image into a short video clip per scene."""

    def __init__(self, config: Optional[Config] = None, backend: str = "svd"):
        self.config = config or Config()
        self.logger = self.config.get_logger("video_generator")
        if backend != "svd":
            raise NotImplementedError(
                f"backend={backend!r} is not implemented yet; only 'svd' is available"
            )
        self.backend = backend
        self.pipe = None
        self._motion_controller = None  # lazily constructed; see _default_motion_controller

    def load_model(self) -> None:
        if self.pipe is not None:
            return
        if StableVideoDiffusionPipeline is None:
            raise ImportError("diffusers is not installed")

        model_id = self.config.svd_model
        with timer(f"loading SVD model '{model_id}'", self.logger):
            dtype = torch.float16 if (self.config.use_fp16 and self.config.device == "cuda") else torch.float32
            kwargs = dict(torch_dtype=dtype, token=self.config.hf_token)
            if self.config.use_fp16 and self.config.device == "cuda":
                kwargs["variant"] = "fp16"
            self.pipe = StableVideoDiffusionPipeline.from_pretrained(model_id, **kwargs)

            if self.config.device == "cuda":
                self.pipe.enable_model_cpu_offload()
                if self.config.svd_enable_forward_chunking:
                    self.pipe.unet.enable_forward_chunking()
            else:
                self.pipe.to(self.config.device)

        self.logger.info("SVD model loaded (%s)", model_id)

    def _resolve_motion_bucket_id(self, motion_hint: Optional[dict]) -> int:
        if motion_hint and "motion_bucket_id" in motion_hint:
            return motion_hint["motion_bucket_id"]
        return self.config.svd_motion_bucket_id

    def _resolve_noise_aug_strength(self, motion_hint: Optional[dict]) -> float:
        if motion_hint and "noise_aug_strength" in motion_hint:
            return motion_hint["noise_aug_strength"]
        return self.config.svd_noise_aug_strength

    def _default_motion_controller(self):
        if self._motion_controller is None:
            from modules.motion_controller import MotionController

            self._motion_controller = MotionController(self.config)
        return self._motion_controller

    @retry(max_attempts=2, delay_seconds=2.0)
    def _generate(self, keyframe: Image.Image, motion_bucket_id: int, noise_aug_strength: float, seed: Optional[int]):
        generator = None
        if seed is not None and torch is not None:
            # SVD examples use a CPU generator even when the pipe runs on CUDA.
            generator = torch.Generator(device="cpu").manual_seed(seed)

        result = self.pipe(
            keyframe,
            height=self.config.svd_height,
            width=self.config.svd_width,
            num_inference_steps=self.config.svd_num_inference_steps,
            min_guidance_scale=self.config.svd_min_guidance_scale,
            max_guidance_scale=self.config.svd_max_guidance_scale,
            fps=self.config.svd_fps,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            decode_chunk_size=self.config.svd_decode_chunk_size,
            generator=generator,
        )
        return result.frames[0]  # List[PIL.Image.Image]

    def generate_clip_for_scene(
        self,
        scene,
        keyframe_path: Path,
        motion_hint: Optional[dict] = None,
        seed: Optional[int] = None,
        camera_motion_type: Optional[str] = None,
        action_plan=None,
        motion_controller=None,
    ) -> Path:
        """Generate one short clip for a scene and save it as an mp4 under
        config.clips_dir. Returns the clip file path.

        `motion_hint` tunes the SVD call itself (intensity only). Passing
        `camera_motion_type` (any modules.motion_controller.MotionController
        camera- or subject-motion preset name) or `action_plan` (a
        modules.motion_controller.ActionPlan) additionally post-processes
        the rendered frames with real directional motion before encoding —
        `action_plan` takes priority if both are given. A MotionController
        is constructed on demand unless one is passed in explicitly (e.g. to
        reuse a single instance, with one shared shake RNG, across scenes).
        """
        self.load_model()
        keyframe = Image.open(keyframe_path).convert("RGB").resize(
            (self.config.svd_width, self.config.svd_height)
        )
        motion_bucket_id = self._resolve_motion_bucket_id(motion_hint)
        noise_aug_strength = self._resolve_noise_aug_strength(motion_hint)

        with timer(f"SVD inference scene {scene.index}", self.logger):
            frames = self._generate(keyframe, motion_bucket_id, noise_aug_strength, seed)

        if action_plan is not None:
            controller = motion_controller or self._default_motion_controller()
            frames = controller.apply_action_sequence(frames, action_plan)
        elif camera_motion_type is not None:
            controller = motion_controller or self._default_motion_controller()
            frames = controller.apply_named_motion(frames, camera_motion_type)

        out_path = self.config.clips_dir / f"scene_{scene.index:02d}.mp4"
        self._frames_to_mp4(frames, out_path)
        return out_path

    def generate_clips_for_scenes(
        self,
        scenes: List,
        keyframe_paths: List[Path],
        motion_hints: Optional[List[Optional[dict]]] = None,
        camera_motion_types: Optional[List[Optional[str]]] = None,
        action_plans: Optional[List] = None,
        motion_controller=None,
        seed: Optional[int] = None,
    ) -> List[Path]:
        self.load_model()
        n = len(scenes)
        motion_hints = motion_hints or [None] * n
        camera_motion_types = camera_motion_types or [None] * n
        action_plans = action_plans or [None] * n
        clip_paths = []
        with tqdm(total=n, desc="Generating video clips") as pbar:
            for scene, keyframe_path, hint, cam_motion, plan in zip(
                scenes, keyframe_paths, motion_hints, camera_motion_types, action_plans
            ):
                clip_paths.append(
                    self.generate_clip_for_scene(
                        scene,
                        keyframe_path,
                        motion_hint=hint,
                        seed=seed,
                        camera_motion_type=cam_motion,
                        action_plan=plan,
                        motion_controller=motion_controller,
                    )
                )
                clear_gpu_memory()
                pbar.update(1)
        return clip_paths

    def _frames_to_mp4(self, frames: List[Image.Image], out_path: Path) -> None:
        """Encode a PIL frame list to mp4 via moviepy's ImageSequenceClip.

        Avoids round-tripping through diffusers.utils.export_to_video's own
        opencv mp4v encoding (not reliably re-encodable by every moviepy/
        ffmpeg build) — frames are encoded once, directly, with the
        project's standard libx264 codec.
        """
        import numpy as np
        from moviepy.editor import ImageSequenceClip

        arrays = [np.array(f) for f in frames]
        clip = ImageSequenceClip(arrays, fps=self.config.svd_fps)
        clip.write_videofile(
            str(out_path),
            fps=self.config.svd_fps,
            codec=self.config.video_codec,
            audio=False,
            logger=None,
        )
        clip.close()

    def unload(self) -> None:
        self.pipe = None
        clear_gpu_memory()
