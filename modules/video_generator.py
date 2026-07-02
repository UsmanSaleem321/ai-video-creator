"""Image-to-video generation module.

Phase 1 backend: Stable Video Diffusion (SVD / SVD-XT via diffusers). A
future phase is expected to add an AnimateDiff-based backend for
prompt-driven "action sequences" (explicit camera pans, character actions
beyond what single-image SVD conditioning can express).

Design for future extension:
  - `VideoGenerator` is the public-facing class main.py calls; it currently
    hard-wires the SVD backend, but its constructor accepts a `backend`
    argument (only "svd" is implemented; anything else raises
    NotImplementedError) so a later phase can add "animatediff" without
    changing the class's public method signatures.
  - Motion control seam: `generate_clip_for_scene` accepts an optional
    `motion_hint: Optional[dict] = None` parameter that is currently UNUSED
    (this phase always uses the config's static svd_motion_bucket_id). A
    future `modules/motion_controller.py` (not implemented yet) is expected
    to populate `motion_hint` (e.g. {"motion_bucket_id": ...}) per scene
    based on camera-motion presets — see `_resolve_motion_bucket_id` below.
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
        # TODO(motion controller): a future MotionController.add_camera_motion()
        # will populate motion_hint with a per-scene motion_bucket_id derived
        # from a camera-motion preset (e.g. "slow pan" -> lower bucket,
        # "action" -> higher bucket). Until then every scene uses the static
        # config default.
        if motion_hint and "motion_bucket_id" in motion_hint:
            return motion_hint["motion_bucket_id"]
        return self.config.svd_motion_bucket_id

    @retry(max_attempts=2, delay_seconds=2.0)
    def _generate(self, keyframe: Image.Image, motion_bucket_id: int, seed: Optional[int]):
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
            noise_aug_strength=self.config.svd_noise_aug_strength,
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
    ) -> Path:
        """Generate one short clip for a scene and save it as an mp4 under
        config.clips_dir. Returns the clip file path."""
        self.load_model()
        keyframe = Image.open(keyframe_path).convert("RGB").resize(
            (self.config.svd_width, self.config.svd_height)
        )
        motion_bucket_id = self._resolve_motion_bucket_id(motion_hint)

        with timer(f"SVD inference scene {scene.index}", self.logger):
            frames = self._generate(keyframe, motion_bucket_id, seed)

        out_path = self.config.clips_dir / f"scene_{scene.index:02d}.mp4"
        self._frames_to_mp4(frames, out_path)
        return out_path

    def generate_clips_for_scenes(
        self,
        scenes: List,
        keyframe_paths: List[Path],
        motion_hints: Optional[List[Optional[dict]]] = None,
        seed: Optional[int] = None,
    ) -> List[Path]:
        self.load_model()
        motion_hints = motion_hints or [None] * len(scenes)
        clip_paths = []
        with tqdm(total=len(scenes), desc="Generating video clips") as pbar:
            for scene, keyframe_path, hint in zip(scenes, keyframe_paths, motion_hints):
                clip_paths.append(
                    self.generate_clip_for_scene(scene, keyframe_path, motion_hint=hint, seed=seed)
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
