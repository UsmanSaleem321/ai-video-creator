"""Text-to-video generation with CogVideoX.

This module is intentionally self-contained and lazily imports the heavy
video stack so slideshow and SVD modes keep working when CogVideoX-specific
dependencies are not installed.
"""

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from config import Config
from utils.helpers import clear_gpu_memory, timer, retry

try:
    import torch
    from diffusers import CogVideoXPipeline
except ImportError:
    torch = None
    CogVideoXPipeline = None


class CogVideoGenerationError(RuntimeError):
    pass


class CogVideoGenerator:
    """Generate natural text-to-video scene clips with CogVideoX-2b."""

    def __init__(
        self,
        config: Optional[Config] = None,
        quality: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.config = config or Config()
        if quality is not None:
            self.config.quality_preset = quality
            self.config._apply_quality_preset()
        self.logger = self.config.get_logger("cogvideo_generator")
        self.model_name = model_name or self.config.cogvideo_model
        self.pipe = None

    # ------------------------------------------------------------- loading
    def load_model(self) -> None:
        if self.pipe is not None:
            return
        if CogVideoXPipeline is None:
            raise ImportError(
                "CogVideoXPipeline is unavailable. Install diffusers>=0.30.0 "
                "and the CogVideo dependencies from requirements.txt."
            )
        if torch is None:
            raise ImportError("torch is not installed")

        dtype = torch.float16 if (self.config.use_fp16 and self.config.device == "cuda") else torch.float32
        with timer(f"loading CogVideoX model '{self.model_name}'", self.logger):
            self.pipe = CogVideoXPipeline.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                token=self.config.hf_token,
            )

            if self.config.device == "cuda":
                self.pipe.enable_model_cpu_offload()
            else:
                self.pipe.to(self.config.device)

            self._enable_memory_savers()

        self.logger.info("CogVideoX model loaded (%s)", self.model_name)

    def _enable_memory_savers(self) -> None:
        """Turn on best-effort memory savers supported by the installed Diffusers build."""
        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()

        vae = getattr(self.pipe, "vae", None)
        if vae is not None:
            if hasattr(vae, "enable_slicing"):
                vae.enable_slicing()
            if hasattr(vae, "enable_tiling"):
                vae.enable_tiling()

    # ---------------------------------------------------------- generation
    def _preset(self) -> dict:
        preset = self.config.cogvideo_quality_presets.get(
            self.config.quality_preset,
            self.config.cogvideo_quality_presets["balanced"],
        )
        return dict(preset)

    def _resolve_generation_settings(
        self,
        duration: Optional[float],
        width: Optional[int],
        height: Optional[int],
        num_frames: Optional[int],
        num_inference_steps: Optional[int],
        guidance_scale: Optional[float],
    ) -> dict:
        preset = self._preset()
        fps = int(preset["fps"])
        resolved_duration = float(duration or preset["duration"])
        resolved_frames = int(num_frames or round(resolved_duration * fps))
        return {
            "duration": resolved_duration,
            "num_frames": max(1, resolved_frames),
            "num_inference_steps": int(num_inference_steps or preset["num_inference_steps"]),
            "guidance_scale": float(guidance_scale or preset["guidance_scale"]),
            "width": int(width or preset["width"]),
            "height": int(height or preset["height"]),
            "fps": fps,
        }

    @retry(max_attempts=2, delay_seconds=2.0)
    def _run_pipeline(self, prompt: str, settings: dict, seed: Optional[int]):
        generator = None
        if seed is not None and torch is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        kwargs = dict(
            prompt=prompt,
            negative_prompt=self.config.cogvideo_negative_prompt,
            height=settings["height"],
            width=settings["width"],
            num_frames=settings["num_frames"],
            num_inference_steps=settings["num_inference_steps"],
            guidance_scale=settings["guidance_scale"],
            generator=generator,
        )
        try:
            return self.pipe(**kwargs, output_type="pil").frames[0]
        except TypeError:
            return self.pipe(**kwargs).frames[0]

    def generate_video(
        self,
        prompt: str,
        duration: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate a short clip and return frames as uint8 RGB arrays.

        The returned shape is (frames, height, width, 3). Duration defaults to
        the selected quality preset, but callers may request any 2-5 second
        value that fits their GPU budget.
        """
        self.load_model()
        settings = self._resolve_generation_settings(
            duration, width, height, num_frames, num_inference_steps, guidance_scale
        )
        with timer(
            f"CogVideoX inference ({settings['num_frames']} frames, "
            f"{settings['width']}x{settings['height']})",
            self.logger,
        ):
            frames = self._run_pipeline(prompt, settings, seed)

        arrays = self._frames_to_arrays(frames)
        clear_gpu_memory()
        return arrays

    def generate_scene_videos(
        self,
        scenes: Sequence,
        style: str = "cinematic",
        characters: Optional[Sequence] = None,
        duration: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> List[Path]:
        """Generate and save one CogVideoX MP4 clip per story scene."""
        self.load_model()
        clip_paths: List[Path] = []
        character_context = self._build_character_context(characters)
        with tqdm(total=len(scenes), desc="Generating CogVideoX clips") as pbar:
            for scene in scenes:
                prompt = self._build_scene_prompt(scene, style, character_context)
                try:
                    frames = self.generate_video(prompt, duration=duration, seed=seed)
                    out_path = self.config.clips_dir / f"scene_{scene.index:02d}_cogvideo.mp4"
                    self.save_video(frames, out_path)
                    clip_paths.append(out_path)
                except Exception as exc:  # noqa: BLE001
                    clear_gpu_memory()
                    raise CogVideoGenerationError(
                        f"CogVideoX failed for scene {scene.index}: {exc}"
                    ) from exc
                finally:
                    pbar.update(1)
        return clip_paths

    # --------------------------------------------------------------- output
    def save_video(self, frames: np.ndarray, output_path: Path, fps: Optional[int] = None) -> Path:
        """Save RGB numpy frames to an MP4 file and return its path."""
        from moviepy.editor import ImageSequenceClip

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = self._normalize_arrays(frames)
        clip = ImageSequenceClip(list(arrays), fps=fps or self.config.cogvideo_fps)
        clip.write_videofile(
            str(output_path),
            fps=fps or self.config.cogvideo_fps,
            codec=self.config.video_codec,
            audio=False,
            logger=None,
        )
        clip.close()
        return output_path

    @staticmethod
    def _frames_to_arrays(frames) -> np.ndarray:
        arrays = []
        for frame in frames:
            if isinstance(frame, Image.Image):
                arrays.append(np.array(frame.convert("RGB"), dtype=np.uint8))
            else:
                arrays.append(np.asarray(frame, dtype=np.uint8))
        return CogVideoGenerator._normalize_arrays(np.stack(arrays, axis=0))

    @staticmethod
    def _normalize_arrays(frames: np.ndarray) -> np.ndarray:
        frames = np.asarray(frames)
        if frames.dtype != np.uint8:
            if frames.max() <= 1.0:
                frames = frames * 255.0
            frames = np.clip(frames, 0, 255).astype(np.uint8)
        if frames.ndim != 4 or frames.shape[-1] not in (3, 4):
            raise CogVideoGenerationError(
                f"Expected frames with shape (n, h, w, 3/4), got {frames.shape}"
            )
        if frames.shape[-1] == 4:
            frames = frames[..., :3]
        return frames

    @staticmethod
    def _build_character_context(characters: Optional[Sequence]) -> str:
        if not characters:
            return ""
        parts = []
        for character in characters:
            name = getattr(character, "name", "")
            description = getattr(character, "description", "")
            if name or description:
                parts.append(f"{name}: {description}".strip(": "))
        if not parts:
            return ""
        return "Recurring characters, keep identity consistent across shots: " + "; ".join(parts)

    @staticmethod
    def _build_scene_prompt(scene, style: str, character_context: str) -> str:
        visual = getattr(scene, "visual", str(scene))
        motion = getattr(scene, "motion", "none")
        prompt_parts = [
            visual,
            character_context,
            "natural cinematic motion, realistic physics, fluid camera movement",
            "consistent lighting, coherent subject motion, filmic composition",
            f"{style} style",
        ]
        if motion and motion != "none":
            prompt_parts.append(f"motion direction: {motion.replace('_', ' ')}")
        return ", ".join(part for part in prompt_parts if part)

    def unload(self) -> None:
        self.pipe = None
        clear_gpu_memory()
