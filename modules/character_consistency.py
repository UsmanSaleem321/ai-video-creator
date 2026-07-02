"""Character reference generation & IP-Adapter setup for consistent character
rendering across scenes (SD1.5 + IP-Adapter reference portraits).

This module owns:
  - turning `Character` objects (from story_generator) into saved reference
    PIL images on disk (one portrait per character), and
  - attaching the IP-Adapter weights on a caller-supplied
    `StableDiffusionPipeline` instance (reuses ImageGenerator's already-loaded
    pipe; this module does NOT load its own SD1.5 pipeline instance, to avoid
    a second multi-GB model residency on top of the story/image pipeline).

A future ControlNet/AnimateDiff-based consistency path is expected to reuse
`CharacterReferenceStore`'s cached reference images rather than re-deriving
character portraits from scratch.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from config import Config
from utils.helpers import slugify, timer
from utils.prompts import build_character_portrait_prompt
from modules.story_generator import Character


class CharacterConsistencyError(RuntimeError):
    pass


@dataclass
class CharacterReference:
    character: Character
    image_path: Path


class CharacterReferenceStore:
    """Generates and caches one reference portrait per Character."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = self.config.get_logger("character_consistency")
        self._refs: Dict[str, CharacterReference] = {}
        self._ip_adapter_attached = False
        self._neutral_reference: Optional[Image.Image] = None

    def generate_references(self, pipe, characters: List[Character]) -> Dict[str, CharacterReference]:
        """Render one portrait per character using an already-loaded SD1.5
        `pipe` (the same StableDiffusionPipeline instance ImageGenerator
        uses — passed in, not owned, so the caller controls load/unload
        lifecycle).

        Call this BEFORE attach_ip_adapter(): these are plain text-to-image
        portraits, no image conditioning needed for the reference itself.
        """
        characters = characters[: self.config.max_characters]
        for character in characters:
            prompt = build_character_portrait_prompt(character.description)
            with timer(f"generating character reference: {character.name}", self.logger):
                result = pipe(
                    prompt=prompt,
                    negative_prompt=self.config.image_negative_prompt,
                    width=self.config.character_ref_width,
                    height=self.config.character_ref_height,
                    num_inference_steps=self.config.character_ref_steps,
                    guidance_scale=self.config.image_guidance_scale,
                )
            image = result.images[0]
            out_path = self.config.characters_dir / f"{slugify(character.name, max_length=40)}.png"
            image.save(out_path)
            self._refs[character.name.lower()] = CharacterReference(character=character, image_path=out_path)
        return self._refs

    def attach_ip_adapter(self, pipe) -> None:
        """Load IP-Adapter weights onto `pipe` (call once, after
        generate_references(), before generating conditioned keyframes)."""
        if self._ip_adapter_attached:
            return
        pipe.load_ip_adapter(
            self.config.ip_adapter_repo,
            subfolder=self.config.ip_adapter_subfolder,
            weight_name=self.config.ip_adapter_weight_name,
            token=self.config.hf_token,
        )
        pipe.set_ip_adapter_scale(self.config.ip_adapter_scale)
        self._ip_adapter_attached = True

    def references_for_scene(self, scene) -> List[Image.Image]:
        """Return reference PIL image(s) for characters whose name appears
        in this scene's visual/narration text (whole-word, case-insensitive
        match). Capped at one match per scene: passing multiple reference
        images to a single IP-Adapter call has diffusers API shape nuances
        (nested vs. flat lists) that aren't worth the added fragility for a
        first pass, so multi-character scenes get the first-matched
        character's identity conditioning only.
        """
        text = f"{scene.visual} {scene.narration}".lower()
        for name_lower, ref in self._refs.items():
            if re.search(rf"\b{re.escape(name_lower)}\b", text):
                return [Image.open(ref.image_path).convert("RGB")]
        return []

    def _get_neutral_reference(self) -> Image.Image:
        """A plain, content-free placeholder image, used to satisfy
        IP-Adapter's per-call image requirement (see
        resolve_ip_adapter_conditioning) for scenes with no matched
        character. Its content is irrelevant since it's always paired with
        scale=0.0, so it's generated once in-memory rather than saved to
        disk or rendered by the diffusion model.
        """
        if self._neutral_reference is None:
            self._neutral_reference = Image.new(
                "RGB", (self.config.character_ref_width, self.config.character_ref_height), (128, 128, 128)
            )
        return self._neutral_reference

    def resolve_ip_adapter_conditioning(self, scene) -> Tuple[Image.Image, float]:
        """Return (ip_adapter_image, ip_adapter_scale) to pass to the SD
        pipeline for this scene's keyframe generation.

        Once `pipe.load_ip_adapter()` has been called, the UNet's
        cross-attention processors are patched pipe-wide to require image
        embeddings on every subsequent call — diffusers only builds
        `added_cond_kwargs` when `ip_adapter_image` (or
        `ip_adapter_image_embeds`) is passed to that call, so omitting it
        for an "unconditioned" scene crashes deep inside
        `UNet2DConditionModel.process_encoder_hidden_states()` with
        `added_cond_kwargs` being `None`. To avoid that, callers must
        ALWAYS pass some `ip_adapter_image` once IP-Adapter is attached; for
        scenes with no matched character we return an inert neutral
        placeholder together with scale=0.0, which has zero effect on the
        generated image while still satisfying the API contract.
        """
        matches = self.references_for_scene(scene)
        if matches:
            return matches[0], self.config.ip_adapter_scale
        return self._get_neutral_reference(), 0.0

    def unload(self) -> None:
        self._refs.clear()
        self._ip_adapter_attached = False
        self._neutral_reference = None
