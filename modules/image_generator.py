"""Image generation module built on Stable Diffusion 1.5.

Turns a scene's visual description into a rendered image, with batching,
negative prompts, and memory-optimization toggles (attention/vae slicing,
fp16, CPU offload) suited to Colab's typically constrained GPUs.
"""

from pathlib import Path
from typing import List, Optional

from tqdm.auto import tqdm

from config import Config
from utils.helpers import clear_gpu_memory, timer, retry
from utils.prompts import build_image_prompt

try:
    import torch
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
except ImportError:
    torch = None
    StableDiffusionPipeline = DPMSolverMultistepScheduler = None


class ImageGenerationError(RuntimeError):
    pass


class ImageGenerator:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = self.config.get_logger("image_generator")
        self.pipe = None

    def load_model(self) -> None:
        if self.pipe is not None:
            return
        if StableDiffusionPipeline is None:
            raise ImportError("diffusers is not installed")

        with timer(f"loading image model '{self.config.image_model}'", self.logger):
            dtype = torch.float16 if (self.config.use_fp16 and self.config.device == "cuda") else torch.float32
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.config.image_model,
                torch_dtype=dtype,
                safety_checker=None,
                token=self.config.hf_token,
            )
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )

            if self.config.device == "cuda":
                self.pipe = self.pipe.to("cuda")
                if self.config.image_enable_attention_slicing:
                    self.pipe.enable_attention_slicing()
                if self.config.image_enable_vae_slicing:
                    self.pipe.enable_vae_slicing()
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                except Exception:
                    self.logger.info("xformers not available, continuing without it")
            else:
                self.pipe.enable_attention_slicing()

        self.logger.info("Image model loaded on %s", self.config.device)

    @retry(max_attempts=2, delay_seconds=2.0)
    def _generate_batch(self, prompts: List[str], negative_prompt: str, seed: Optional[int]):
        generator = None
        if seed is not None and torch is not None:
            generator = torch.Generator(device=self.config.device).manual_seed(seed)

        with torch.autocast(self.config.device) if self.config.device == "cuda" else _null_context():
            result = self.pipe(
                prompt=prompts,
                negative_prompt=[negative_prompt] * len(prompts),
                width=self.config.image_width,
                height=self.config.image_height,
                num_inference_steps=self.config.image_num_inference_steps,
                guidance_scale=self.config.image_guidance_scale,
                generator=generator,
            )
        return result.images

    def generate_images_for_scenes(
        self,
        scenes: List,
        style: str = "cinematic",
        seed: Optional[int] = None,
    ) -> List[Path]:
        """Generate one image per scene, batching according to config.image_batch_size."""
        self.load_model()
        image_paths: List[Path] = []
        batch_size = max(1, self.config.image_batch_size)
        negative_prompt = self.config.image_negative_prompt

        prompts = [
            build_image_prompt(scene.visual, style=style, extra_suffix=self.config.image_style_suffix)
            for scene in scenes
        ]

        with tqdm(total=len(scenes), desc="Generating images") as pbar:
            for batch_start in range(0, len(prompts), batch_size):
                batch_prompts = prompts[batch_start: batch_start + batch_size]
                batch_scenes = scenes[batch_start: batch_start + batch_size]

                try:
                    images = self._generate_batch(batch_prompts, negative_prompt, seed)
                except Exception as exc:  # noqa: BLE001
                    self.logger.error("Batch generation failed, falling back to solo generation: %s", exc)
                    images = []
                    for p in batch_prompts:
                        images.extend(self._generate_batch([p], negative_prompt, seed))

                for scene, image in zip(batch_scenes, images):
                    out_path = self.config.images_dir / f"scene_{scene.index:02d}.png"
                    image.save(out_path)
                    image_paths.append(out_path)
                    pbar.update(1)

                clear_gpu_memory()

        return image_paths

    def generate_keyframes_for_scenes(
        self,
        scenes: List,
        style: str = "cinematic",
        character_store=None,
        extra_prompt_suffixes: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> List[Path]:
        """Like generate_images_for_scenes, but conditions each scene on a
        matching character reference (via IP-Adapter) when `character_store`
        is provided. Produces one keyframe PNG per scene at
        config.image_width x config.image_height (resized later for SVD).

        `extra_prompt_suffixes`, if given, is a list index-aligned with
        `scenes` — e.g. an action preset's descriptive text (see
        modules.motion_controller.ActionPlan.prompt_suffix) — appended to
        that scene's prompt so the keyframe itself depicts the action the
        video-generation stage will later add camera motion to.

        Caller is responsible for calling character_store.generate_references()
        and character_store.attach_ip_adapter(self.pipe) BEFORE calling this
        method (kept separate from generate_images_for_scenes so the
        slideshow path's method stays untouched).
        """
        self.load_model()
        keyframe_paths: List[Path] = []
        negative_prompt = self.config.image_negative_prompt
        extra_prompt_suffixes = extra_prompt_suffixes or [""] * len(scenes)

        with tqdm(total=len(scenes), desc="Generating keyframes") as pbar:
            for scene, extra_suffix in zip(scenes, extra_prompt_suffixes):
                style_suffix = self.config.image_style_suffix
                if extra_suffix:
                    style_suffix = f"{style_suffix}, {extra_suffix}" if style_suffix else extra_suffix
                prompt = build_image_prompt(scene.visual, style=style, extra_suffix=style_suffix)
                ip_images = character_store.references_for_scene(scene) if character_store else []

                call_kwargs = dict(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=self.config.image_width,
                    height=self.config.image_height,
                    num_inference_steps=self.config.image_num_inference_steps,
                    guidance_scale=self.config.image_guidance_scale,
                )
                if ip_images:
                    call_kwargs["ip_adapter_image"] = ip_images[0]

                with torch.autocast(self.config.device) if self.config.device == "cuda" else _null_context():
                    result = self.pipe(**call_kwargs)

                out_path = self.config.images_dir / f"scene_{scene.index:02d}_keyframe.png"
                result.images[0].save(out_path)
                keyframe_paths.append(out_path)
                pbar.update(1)
                clear_gpu_memory()

        return keyframe_paths

    def generate_preview(self, scene, style: str = "cinematic") -> Path:
        """Fast, low-res preview render for quick iteration before a full-quality pass."""
        self.load_model()
        prompt = build_image_prompt(scene.visual, style=style)
        with torch.autocast(self.config.device) if self.config.device == "cuda" else _null_context():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=self.config.image_negative_prompt,
                width=256,
                height=256,
                num_inference_steps=10,
                guidance_scale=self.config.image_guidance_scale,
            )
        out_path = self.config.images_dir / f"preview_scene_{scene.index:02d}.png"
        result.images[0].save(out_path)
        return out_path

    def unload(self) -> None:
        self.pipe = None
        clear_gpu_memory()


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def generate_scene_images(scenes: List, config: Optional[Config] = None, style: str = "cinematic") -> List[Path]:
    generator = ImageGenerator(config)
    try:
        return generator.generate_images_for_scenes(scenes, style=style)
    finally:
        generator.unload()
