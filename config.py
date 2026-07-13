"""
Central configuration for the AI Video Creator pipeline.

All modules import a shared `Config` instance from here so that paths,
model names, and generation parameters stay consistent across the
story, image, audio, and video-assembly stages.
"""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import torch
except ImportError:
    torch = None


def _detect_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _gpu_vram_gb() -> float:
    """Return total VRAM of device 0 in GB, or 0.0 if no GPU is present."""
    if torch is not None and torch.cuda.is_available():
        try:
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            return 0.0
    return 0.0


@dataclass
class Config:
    # ---------------------------------------------------------------- paths
    base_dir: Path = Path(os.getenv("AVC_BASE_DIR", Path(__file__).resolve().parent))
    output_dir: Path = field(init=False)
    scenes_dir: Path = field(init=False)
    images_dir: Path = field(init=False)
    characters_dir: Path = field(init=False)
    clips_dir: Path = field(init=False)
    audio_dir: Path = field(init=False)
    narration_dir: Path = field(init=False)
    music_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)

    # --------------------------------------------------------------- device
    device: str = field(default_factory=_detect_device)
    use_fp16: bool = True
    gpu_vram_gb: float = field(default_factory=_gpu_vram_gb)
    low_vram_threshold_gb: float = 12.0  # below this, fall back to smaller models

    # ------------------------------------------------------------- story LLM
    story_model_primary: str = "NeuralNovel/Mistral-7B-Instruct-v0.2-Neural-Story"
    story_model_fallbacks: tuple = (
        "mistralai/Mistral-7B-Instruct-v0.2",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "gpt2",
    )
    story_max_new_tokens: int = 1500
    story_tokens_per_scene: int = 220  # used to scale the token budget for large num_scenes
    story_temperature: float = 0.85
    story_top_p: float = 0.92
    story_num_scenes: int = 6  # target within the 5-7 range
    story_load_in_4bit: bool = True

    # -------------------------------------------------------- image (SD 1.5)
    image_model: str = "runwayml/stable-diffusion-v1-5"
    image_width: int = 512
    image_height: int = 512
    image_num_inference_steps: int = 30
    image_guidance_scale: float = 7.5
    image_negative_prompt: str = (
        "blurry, low quality, low resolution, distorted, deformed, disfigured, "
        "bad anatomy, watermark, text, signature, cropped, out of frame, "
        "extra limbs, worst quality, jpeg artifacts"
    )
    image_batch_size: int = 1
    image_enable_attention_slicing: bool = True
    image_enable_vae_slicing: bool = True
    image_style_suffix: str = "cinematic lighting, highly detailed, 8k, concept art"

    # --------------------------------------------------------------- Bark TTS
    bark_model: str = "suno/bark"
    bark_voice_preset: str = "v2/en_speaker_6"
    bark_text_temp: float = 0.7
    bark_waveform_temp: float = 0.7

    # --------------------------------------------------------- Stable Audio
    music_model: str = "stabilityai/stable-audio-open-1.0"
    music_duration_seconds: float = 45.0
    music_steps: int = 100
    music_cfg_scale: float = 7.0

    # ------------------------------------------------------------- video
    scene_duration_seconds: float = 5.0
    transition_duration_seconds: float = 1.0
    fps: int = 30
    video_width: int = 1920
    video_height: int = 1080
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    music_volume: float = 0.25  # 0.2 - 0.3
    narration_volume: float = 1.0
    subtitle_font: str = "DejaVu-Sans-Bold"
    subtitle_fontsize: int = 42
    subtitle_color: str = "white"
    subtitle_stroke_color: str = "black"
    subtitle_stroke_width: int = 2

    # ------------------------------------------------------------- quality
    quality_preset: str = "balanced"  # "draft" | "balanced" | "high"

    # ------------------------------------------------------------- pipeline mode
    pipeline_mode: str = "slideshow"  # "slideshow" (static images) | "cinematic" (real motion clips)

    # --------------------------------------------------- character consistency (IP-Adapter)
    ip_adapter_repo: str = "h94/IP-Adapter"
    ip_adapter_subfolder: str = "models"
    ip_adapter_weight_name: str = "ip-adapter_sd15.bin"
    ip_adapter_scale: float = 0.6  # 0.0-1.0; balances identity vs prompt adherence
    character_ref_width: int = 512
    character_ref_height: int = 512
    character_ref_steps: int = 30
    max_characters: int = 3  # cap on character reference portraits generated per story

    # ------------------------------------------------------- video generation (SVD)
    svd_model: str = "stabilityai/stable-video-diffusion-img2vid-xt"
    svd_model_fallback: str = "stabilityai/stable-video-diffusion-img2vid"  # 14-frame, lighter
    svd_width: int = 1024
    svd_height: int = 576  # SVD's trained resolution; keyframes are resized to this before SVD
    svd_num_frames: int = 25  # informational (clip-duration estimation); not passed to the pipeline call
    svd_num_inference_steps: int = 25
    svd_fps: int = 7  # SVD's native/trained fps
    svd_motion_bucket_id: int = 100  # below the diffusers default (127) for subtler, controlled motion
    svd_noise_aug_strength: float = 0.02
    svd_decode_chunk_size: int = 4  # conservative for T4; lower = less VRAM, more flicker risk
    svd_min_guidance_scale: float = 1.0
    svd_max_guidance_scale: float = 3.0
    svd_enable_forward_chunking: bool = True

    # ------------------------------------------------------ video generation (CogVideoX)
    cogvideo_model: str = "THUDM/CogVideoX-2b"
    cogvideo_enable_fallback: bool = True
    cogvideo_negative_prompt: str = (
        "low quality, blurry, distorted, flickering, jittery, watermark, text, "
        "logo, bad anatomy, deformed, disfigured, inconsistent character"
    )
    cogvideo_quality_presets: dict = field(
        default_factory=lambda: {
            "draft": {
                "duration": 2,
                "num_frames": 16,
                "num_inference_steps": 25,
                "guidance_scale": 5.5,
                "width": 512,
                "height": 512,
                "fps": 8,
            },
            "balanced": {
                "duration": 3,
                "num_frames": 24,
                "num_inference_steps": 50,
                "guidance_scale": 6.0,
                "width": 768,
                "height": 768,
                "fps": 8,
            },
            "high": {
                "duration": 5,
                "num_frames": 40,
                "num_inference_steps": 50,
                "guidance_scale": 6.5,
                "width": 1024,
                "height": 1024,
                "fps": 8,
            },
        }
    )
    cogvideo_duration_seconds: int = 3
    cogvideo_num_frames: int = 24
    cogvideo_num_inference_steps: int = 50
    cogvideo_guidance_scale: float = 6.0
    cogvideo_width: int = 768
    cogvideo_height: int = 768
    cogvideo_fps: int = 8

    # ------------------------------------------------- cinematic transitions
    cinematic_transition_zoom_amount: float = 0.06  # fractional zoom over the transition window

    # ------------------------------------------------------------- logging
    log_level: str = os.getenv("AVC_LOG_LEVEL", "INFO")

    hf_token: Optional[str] = field(default_factory=lambda: os.getenv("HF_TOKEN"))

    def __post_init__(self):
        self.base_dir = Path(self.base_dir)
        self.output_dir = Path(os.getenv("AVC_OUTPUT_DIR", self.base_dir / "output"))
        self.scenes_dir = self.output_dir / "scenes"
        self.images_dir = self.output_dir / "images"
        self.characters_dir = self.output_dir / "characters"
        self.clips_dir = self.output_dir / "clips"
        self.audio_dir = self.output_dir / "audio"
        self.narration_dir = self.audio_dir / "narration"
        self.music_dir = self.audio_dir / "music"
        self.logs_dir = self.output_dir / "logs"
        self.cache_dir = self.output_dir / ".cache"

        self._apply_quality_preset()
        self._apply_low_vram_fallback()
        self.create_directories()

    def _apply_quality_preset(self):
        """Adjust speed/quality trade-off knobs based on the chosen preset."""
        cogvideo_preset = self.cogvideo_quality_presets.get(
            self.quality_preset, self.cogvideo_quality_presets["balanced"]
        )
        self.cogvideo_duration_seconds = cogvideo_preset["duration"]
        self.cogvideo_num_frames = cogvideo_preset["num_frames"]
        self.cogvideo_num_inference_steps = cogvideo_preset["num_inference_steps"]
        self.cogvideo_guidance_scale = cogvideo_preset["guidance_scale"]
        self.cogvideo_width = cogvideo_preset["width"]
        self.cogvideo_height = cogvideo_preset["height"]
        self.cogvideo_fps = cogvideo_preset["fps"]

        if self.quality_preset == "draft":
            self.image_num_inference_steps = 15
            self.music_steps = 50
            self.fps = 24
            self.video_width, self.video_height = 1280, 720
        elif self.quality_preset == "high":
            self.image_num_inference_steps = 50
            self.music_steps = 150
            self.fps = 30
            self.video_width, self.video_height = 1920, 1080
        # "balanced" keeps the dataclass defaults

    def _apply_low_vram_fallback(self):
        """Downgrade heavy settings automatically on small GPUs, or CPU-only boxes."""
        if self.device == "cpu":
            self.use_fp16 = False
            self.story_load_in_4bit = False
            self.image_num_inference_steps = min(self.image_num_inference_steps, 20)
        elif 0 < self.gpu_vram_gb < self.low_vram_threshold_gb:
            self.story_load_in_4bit = True
            self.image_enable_attention_slicing = True
            self.image_enable_vae_slicing = True
            self.svd_model = self.svd_model_fallback  # switch to the lighter 14-frame base SVD
            self.svd_decode_chunk_size = min(self.svd_decode_chunk_size, 2)
            self.svd_num_inference_steps = min(self.svd_num_inference_steps, 20)
            self.cogvideo_num_inference_steps = min(self.cogvideo_num_inference_steps, 25)
            self.cogvideo_width = min(self.cogvideo_width, 512)
            self.cogvideo_height = min(self.cogvideo_height, 512)
            self.cogvideo_num_frames = min(self.cogvideo_num_frames, 16)

    def create_directories(self):
        for d in (
            self.output_dir,
            self.scenes_dir,
            self.images_dir,
            self.characters_dir,
            self.clips_dir,
            self.audio_dir,
            self.narration_dir,
            self.music_dir,
            self.logs_dir,
            self.cache_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def get_logger(self, name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            logger.setLevel(self.log_level)
            fmt = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(fmt)
            logger.addHandler(stream_handler)

            file_handler = logging.FileHandler(self.logs_dir / "pipeline.log")
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)
        return logger


config = Config()
