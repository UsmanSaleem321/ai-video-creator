"""Audio generation module: Bark narration + Stable Audio Open music/SFX.

Two independent sub-generators live here because they have very different
lifecycles (Bark runs once per scene; Stable Audio Open runs once per video)
and different failure modes, but sharing a module keeps all "audio" concerns
in one place per the project's module layout.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm.auto import tqdm

from config import Config
from utils.helpers import clear_gpu_memory, timer, retry
from utils.prompts import build_music_prompt

try:
    import torch
except ImportError:
    torch = None

try:
    import soundfile as sf
except ImportError:
    sf = None


class AudioGenerationError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Narration (Bark)
# --------------------------------------------------------------------------
class NarrationGenerator:
    """Text-to-speech narration using Bark."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = self.config.get_logger("narration_generator")
        self._backend = None  # "bark_pkg" or "transformers"
        self._model = None
        self._processor = None
        self._sample_rate = 24000

    def load_model(self) -> None:
        if self._backend is not None:
            return
        try:
            from bark import preload_models, SAMPLE_RATE  # type: ignore

            with timer("loading Bark (suno-bark package)", self.logger):
                preload_models()
            self._sample_rate = SAMPLE_RATE
            self._backend = "bark_pkg"
            self.logger.info("Loaded Bark via the `bark` package")
            return
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("`bark` package unavailable (%s); trying transformers BarkModel", exc)

        try:
            from transformers import AutoProcessor, BarkModel

            with timer(f"loading Bark model '{self.config.bark_model}' via transformers", self.logger):
                self._processor = AutoProcessor.from_pretrained(self.config.bark_model)
                self._model = BarkModel.from_pretrained(
                    self.config.bark_model,
                    torch_dtype=torch.float16 if self.config.use_fp16 and self.config.device == "cuda" else torch.float32,
                )
                if self.config.device == "cuda":
                    self._model = self._model.to("cuda")
                    self._model.enable_cpu_offload()
            self._sample_rate = self._model.generation_config.sample_rate
            self._backend = "transformers"
            self.logger.info("Loaded Bark via transformers BarkModel")
        except Exception as exc:  # noqa: BLE001
            raise AudioGenerationError(f"Failed to load Bark through any backend: {exc}") from exc

    @retry(max_attempts=2, delay_seconds=2.0)
    def _synthesize(self, text: str) -> np.ndarray:
        if self._backend == "bark_pkg":
            from bark import generate_audio  # type: ignore

            return generate_audio(
                text,
                history_prompt=self.config.bark_voice_preset,
                text_temp=self.config.bark_text_temp,
                waveform_temp=self.config.bark_waveform_temp,
            )

        inputs = self._processor(text, voice_preset=self.config.bark_voice_preset)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            audio_array = self._model.generate(**inputs)
        return audio_array.cpu().numpy().squeeze()

    def generate_narration_for_scenes(self, scenes: List) -> List[Path]:
        self.load_model()
        paths: List[Path] = []
        for scene in tqdm(scenes, desc="Generating narration"):
            audio = self._synthesize(scene.narration)
            out_path = self.config.narration_dir / f"scene_{scene.index:02d}.wav"
            sf.write(out_path, audio, self._sample_rate)
            paths.append(out_path)
            clear_gpu_memory()
        return paths

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._backend = None
        clear_gpu_memory()


# --------------------------------------------------------------------------
# Background music / SFX (Stable Audio Open)
# --------------------------------------------------------------------------
class MusicGenerator:
    """Ambient background music / soundscape generation via Stable Audio Open."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = self.config.get_logger("music_generator")
        self.pipe = None

    def load_model(self) -> None:
        if self.pipe is not None:
            return
        try:
            from diffusers import StableAudioPipeline

            with timer(f"loading music model '{self.config.music_model}'", self.logger):
                dtype = torch.float16 if (self.config.use_fp16 and self.config.device == "cuda") else torch.float32
                self.pipe = StableAudioPipeline.from_pretrained(
                    self.config.music_model, torch_dtype=dtype, token=self.config.hf_token
                )
                if self.config.device == "cuda":
                    self.pipe = self.pipe.to("cuda")
        except Exception as exc:  # noqa: BLE001
            raise AudioGenerationError(
                f"Failed to load Stable Audio Open (requires diffusers>=0.30): {exc}"
            ) from exc

    @retry(max_attempts=2, delay_seconds=2.0)
    def _generate(self, prompt: str, duration: float):
        generator = torch.Generator(self.config.device).manual_seed(0) if torch is not None else None
        result = self.pipe(
            prompt=prompt,
            negative_prompt="low quality, distorted, harsh, silence",
            num_inference_steps=self.config.music_steps,
            audio_end_in_s=duration,
            num_waveforms_per_prompt=1,
            generator=generator,
        )
        return result.audios[0]

    def generate_music(self, theme: str, duration: Optional[float] = None) -> Path:
        duration = duration or self.config.music_duration_seconds
        try:
            self.load_model()
            prompt = build_music_prompt(theme)
            with timer("music generation inference", self.logger):
                output = self._generate(prompt, duration)
            audio = output.T.float().cpu().numpy() if torch is not None and hasattr(output, "cpu") else np.asarray(output).T
            out_path = self.config.music_dir / "background_music.wav"
            sf.write(out_path, audio, 44100)
            return out_path
        except AudioGenerationError as exc:
            self.logger.error("Music generation unavailable, using silent placeholder: %s", exc)
            return self._silent_fallback(duration)

    def _silent_fallback(self, duration: float) -> Path:
        """Write silence so the video pipeline can still run end-to-end without music."""
        sample_rate = 44100
        silence = np.zeros(int(sample_rate * duration), dtype=np.float32)
        out_path = self.config.music_dir / "background_music.wav"
        sf.write(out_path, silence, sample_rate)
        return out_path

    def unload(self) -> None:
        self.pipe = None
        clear_gpu_memory()


def generate_narration_audio(scenes: List, config: Optional[Config] = None) -> List[Path]:
    generator = NarrationGenerator(config)
    try:
        return generator.generate_narration_for_scenes(scenes)
    finally:
        generator.unload()


def generate_background_music(theme: str, config: Optional[Config] = None, duration: Optional[float] = None) -> Path:
    generator = MusicGenerator(config)
    try:
        return generator.generate_music(theme, duration=duration)
    finally:
        generator.unload()
