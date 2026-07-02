"""Story generation module.

Uses an instruction-tuned Mistral-7B variant to turn a short theme string
into a structured, multi-scene cinematic story. Falls back to progressively
smaller models when GPU memory is insufficient to load the primary model.
"""

import re
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

from tqdm.auto import tqdm

from config import Config
from utils.helpers import clear_gpu_memory, timer, retry, save_json
from utils.prompts import STORY_SYSTEM_PROMPT, build_story_prompt

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ImportError:
    torch = None
    AutoModelForCausalLM = AutoTokenizer = BitsAndBytesConfig = None


@dataclass
class Scene:
    index: int
    visual: str
    narration: str

    def to_dict(self):
        return asdict(self)


@dataclass
class Character:
    name: str
    description: str  # physical/visual description suitable for an SD prompt

    def to_dict(self):
        return asdict(self)


class StoryGenerationError(RuntimeError):
    """Raised when no story model (including all fallbacks) could produce output."""


class StoryGenerator:
    """Loads a causal LM and generates a structured, scene-by-scene story."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = self.config.get_logger("story_generator")
        self.model = None
        self.tokenizer = None
        self.loaded_model_name: Optional[str] = None

    # ------------------------------------------------------------- loading
    def _candidate_models(self) -> List[str]:
        return [self.config.story_model_primary, *self.config.story_model_fallbacks]

    def _build_quantization_config(self):
        if not self.config.story_load_in_4bit or self.config.device != "cuda":
            return None
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    def load_model(self) -> None:
        """Try the primary model first, then fall back down the list.

        Each candidate is attempted with 4-bit quantization (when on a CUDA
        device) before moving on; this mirrors the config's low-VRAM
        handling and keeps a single 15GB Colab GPU able to run the 7B model.
        """
        if self.model is not None:
            return

        last_error: Optional[Exception] = None
        for model_name in self._candidate_models():
            try:
                with timer(f"loading story model '{model_name}'", self.logger):
                    self._load_specific_model(model_name)
                self.loaded_model_name = model_name
                self.logger.info("Successfully loaded story model: %s", model_name)
                return
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Failed to load '%s': %s", model_name, exc)
                last_error = exc
                self.model = None
                self.tokenizer = None
                clear_gpu_memory()

        raise StoryGenerationError(
            f"All story model candidates failed to load. Last error: {last_error}"
        )

    def _load_specific_model(self, model_name: str) -> None:
        if AutoModelForCausalLM is None:
            raise ImportError("transformers is not installed")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=self.config.hf_token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_config = self._build_quantization_config()
        load_kwargs = dict(token=self.config.hf_token)
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = (
                torch.float16 if (self.config.use_fp16 and self.config.device == "cuda") else torch.float32
            )

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if quant_config is None and self.config.device == "cuda":
            self.model = self.model.to(self.config.device)
        self.model.eval()

    # ---------------------------------------------------------- generation
    @retry(max_attempts=2, delay_seconds=1.0)
    def _generate_raw_text(self, prompt: str, max_new_tokens: int) -> str:
        messages = [
            {"role": "system", "content": STORY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self.model.device)
        except Exception:
            # Base models (e.g. gpt2) without a chat template.
            full_prompt = f"{STORY_SYSTEM_PROMPT}\n\n{prompt}"
            input_ids = self.tokenizer(full_prompt, return_tensors="pt").input_ids.to(
                self.model.device
            )

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=self.config.story_temperature,
                top_p=self.config.story_top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = output_ids[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def generate_story(self, theme: str, num_scenes: Optional[int] = None) -> List[Scene]:
        """Generate a structured story and return it as a list of Scene objects.

        Thin wrapper around generate_story_with_characters() that drops the
        character sheet, kept so the slideshow pipeline's call site and
        return type never change.
        """
        scenes, _characters = self.generate_story_with_characters(theme, num_scenes=num_scenes)
        return scenes

    def generate_story_with_characters(
        self, theme: str, num_scenes: Optional[int] = None
    ) -> Tuple[List[Scene], List[Character]]:
        """Generate a structured story plus a short recurring-character sheet.

        Both come from the same generation call (one CHARACTERS block ahead
        of the SCENE blocks) so there's no extra model inference cost versus
        generate_story().
        """
        num_scenes = num_scenes or self.config.story_num_scenes
        self.load_model()

        # Scale the generation budget with the requested scene count so
        # long videos (many scenes) aren't silently truncated mid-story.
        max_new_tokens = max(
            self.config.story_max_new_tokens,
            num_scenes * self.config.story_tokens_per_scene,
        )

        prompt = build_story_prompt(theme, num_scenes)
        with timer("story generation inference", self.logger):
            raw_text = self._generate_raw_text(prompt, max_new_tokens)

        scenes = self._parse_scenes(raw_text)
        characters = self._parse_characters(raw_text)

        if len(scenes) < 3:
            self.logger.warning(
                "Parsed only %d scenes from primary format, retrying with simpler prompt",
                len(scenes),
            )
            simple_prompt = build_story_prompt(theme, num_scenes, simple=True)
            raw_text = self._generate_raw_text(simple_prompt, max_new_tokens)
            scenes = self._parse_scenes(raw_text)
            characters = self._parse_characters(raw_text)

        if not scenes:
            scenes = self._fallback_scenes(theme, num_scenes)

        scenes = scenes[:num_scenes]
        for i, scene in enumerate(scenes, start=1):
            scene.index = i
        characters = characters[: self.config.max_characters]

        save_json(
            {
                "theme": theme,
                "model": self.loaded_model_name,
                "scenes": [s.to_dict() for s in scenes],
                "characters": [c.to_dict() for c in characters],
            },
            self.config.scenes_dir / "story.json",
        )
        return scenes, characters

    # -------------------------------------------------------------- parsing
    def _parse_scenes(self, text: str) -> List[Scene]:
        """Parse `SCENE / VISUAL / NARRATION` blocks out of raw LLM output.

        Tolerant of minor formatting drift (extra whitespace, missing scene
        numbers, markdown bullets) since generated text rarely matches the
        template byte-for-byte.
        """
        pattern = re.compile(
            r"SCENE\s*:?\s*\d*\s*"
            r"VISUAL\s*:?\s*(?P<visual>.*?)\s*"
            r"NARRATION\s*:?\s*(?P<narration>.*?)"
            r"(?=SCENE\s*:?\s*\d*|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
        scenes: List[Scene] = []
        for i, match in enumerate(pattern.finditer(text), start=1):
            visual = self._clean(match.group("visual"))
            narration = self._clean(match.group("narration"))
            if visual and narration:
                scenes.append(Scene(index=i, visual=visual, narration=narration))
        return scenes

    def _parse_characters(self, text: str) -> List[Character]:
        """Parse the `CHARACTERS: Name: description, Name: description` block.

        Tolerant of "none"/absence (returns []) and of the model omitting
        per-character descriptions.
        """
        match = re.search(r"CHARACTERS\s*:?\s*(?P<body>.*?)(?=SCENE\s*:?\s*\d*|\Z)", text, re.IGNORECASE | re.DOTALL)
        if not match:
            return []
        body = self._clean(match.group("body"))
        if not body or body.lower() in {"none", "n/a", "none."}:
            return []

        # Split on commas that precede the next "Name:" entry (not on every
        # comma, since a single character's description often contains
        # commas of its own, e.g. "Aria: a tall, dark-haired pilot").
        entries = re.split(r",\s*(?=[A-Z][\w\s'-]{0,40}:)", body)

        characters: List[Character] = []
        for entry in entries:
            entry = entry.strip()
            if not entry or ":" not in entry:
                continue
            name, description = entry.split(":", 1)
            name = name.strip().strip("*-• ")
            description = description.strip()
            if name and description:
                characters.append(Character(name=name, description=description))
        return characters

    @staticmethod
    def _clean(value: str) -> str:
        value = value.strip().strip("*-• ")
        value = re.sub(r"\s+", " ", value)
        return value

    def _fallback_scenes(self, theme: str, num_scenes: int) -> List[Scene]:
        """Last-resort deterministic scenes if the LLM output is unparseable."""
        self.logger.error("Falling back to template-based scenes for theme: %s", theme)
        scenes = []
        beats = [
            "an establishing wide shot introducing the setting",
            "a closer look revealing the central subject",
            "a moment of tension or change",
            "a turning point in the story",
            "the climax of the scene",
            "the resolution",
            "a final lingering shot",
        ]
        for i in range(num_scenes):
            beat = beats[i % len(beats)]
            scenes.append(
                Scene(
                    index=i + 1,
                    visual=f"{theme}, {beat}, cinematic composition",
                    narration=f"Our story continues as we witness {beat} in a world of {theme}.",
                )
            )
        return scenes

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        clear_gpu_memory()


def generate_story_scenes(theme: str, config: Optional[Config] = None) -> List[Scene]:
    """Convenience function: load model, generate scenes, unload model."""
    generator = StoryGenerator(config)
    try:
        with tqdm(total=1, desc="Generating story") as pbar:
            scenes = generator.generate_story(theme)
            pbar.update(1)
        return scenes
    finally:
        generator.unload()
