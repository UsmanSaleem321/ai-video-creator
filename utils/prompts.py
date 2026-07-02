"""Prompt templates for story generation and image styling.

Kept separate from the modules that use them so that prompt engineering can
be iterated on without touching model-loading/inference code.
"""

STORY_SYSTEM_PROMPT = (
    "You are a professional screenwriter and storyboard artist who writes "
    "short cinematic stories for AI-generated video. You always follow the "
    "requested output format exactly, with no extra commentary."
)

STORY_INSTRUCTION_TEMPLATE = """Write a short cinematic story based on this theme: "{theme}"

First, output ONE character block in exactly this format:

CHARACTERS: <comma-separated list of 0-3 recurring named characters in this story, each formatted as "Name: one-sentence physical description suitable for an image generator", or the single word "none" if there are no recurring named characters>

Then break the story into exactly {num_scenes} scenes. For EACH scene, output a block in exactly this format (plain text, no markdown, no numbering other than shown):

SCENE: <scene number>
VISUAL: <a vivid, concrete visual description of what the camera sees, suitable for an image generation model - focus on subject, setting, lighting, composition, mood, no dialogue>
NARRATION: <one to three sentences of narration text a voice actor would read aloud for this scene, written in an engaging storytelling voice>
MOTION: <exactly one word from this list, whichever best fits the scene: none, pan_left, pan_right, zoom_in, zoom_out, dolly_zoom, orbit, crane_up, crane_down, tracking_shot, dynamic_tracking, walk_forward, run_forward, fly_up, fly_down, jump, rotate, fight, chase, explosion, fly_through>

MOTION guidance: use "none" for a calm/static shot; a camera-motion word (pan_left, pan_right, zoom_in, zoom_out, dolly_zoom, orbit, crane_up, crane_down, tracking_shot, dynamic_tracking) for a shot that needs dramatic camera movement; a subject-motion word (walk_forward, run_forward, fly_up, fly_down, jump, rotate) when a character or object is the one moving; or fight/chase/explosion/fly_through only for a scene that IS that action beat.

Rules:
- Exactly one CHARACTERS block, before SCENE 1.
- Exactly {num_scenes} SCENE blocks, numbered 1 to {num_scenes} in order.
- Keep VISUAL descriptions self-contained (a reader should understand the scene without reading others).
- Keep NARRATION concise: 15-40 words per scene.
- MOTION must be exactly one word from the list above, nothing else.
- Maintain a consistent tone, setting, and characters across all scenes so the story feels like one continuous narrative.
- If a character from the CHARACTERS block appears in a scene, mention them by name in that scene's VISUAL and/or NARRATION text.
- Do not include any text before the CHARACTERS block or after the final scene.
"""

# Fallback template for smaller / less instruction-tuned models that struggle
# with the full instruction above.
STORY_SIMPLE_TEMPLATE = """Theme: {theme}

First write one line:
CHARACTERS: <name: description, ...> or "none"

Write {num_scenes} short scenes for a video. For each scene write:
SCENE: <number>
VISUAL: <what we see>
NARRATION: <what the narrator says>
MOTION: <one word: none, pan_left, pan_right, zoom_in, zoom_out, dolly_zoom, orbit, crane_up, crane_down, tracking_shot, dynamic_tracking, walk_forward, run_forward, fly_up, fly_down, jump, rotate, fight, chase, explosion, or fly_through>
"""

# Visual style modifiers appended to every image prompt to keep art direction
# consistent across the whole video.
IMAGE_STYLE_PRESETS = {
    "cinematic": "cinematic lighting, dramatic composition, film still, highly detailed, 8k, depth of field",
    "concept_art": "digital concept art, matte painting, trending on artstation, highly detailed, dramatic lighting",
    "photorealistic": "photorealistic, ultra detailed, professional photography, sharp focus, 8k uhd",
    "anime": "anime key visual, studio quality, vibrant colors, detailed background",
    "noir": "black and white, film noir style, high contrast, dramatic shadows, moody atmosphere",
}

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, low resolution, distorted, deformed, disfigured, "
    "bad anatomy, extra limbs, mutated hands, watermark, text, signature, "
    "logo, cropped, out of frame, worst quality, jpeg artifacts, duplicate"
)


def build_story_prompt(theme: str, num_scenes: int, simple: bool = False) -> str:
    template = STORY_SIMPLE_TEMPLATE if simple else STORY_INSTRUCTION_TEMPLATE
    return template.format(theme=theme.strip(), num_scenes=num_scenes)


def build_image_prompt(visual_description: str, style: str = "cinematic", extra_suffix: str = "") -> str:
    style_text = IMAGE_STYLE_PRESETS.get(style, IMAGE_STYLE_PRESETS["cinematic"])
    parts = [visual_description.strip(), style_text]
    if extra_suffix:
        parts.append(extra_suffix.strip())
    return ", ".join(p for p in parts if p)


def build_character_portrait_prompt(description: str, extra_suffix: str = "") -> str:
    parts = [
        description.strip(),
        "portrait, consistent character design, plain neutral background, "
        "front-facing, highly detailed, cinematic lighting",
    ]
    if extra_suffix:
        parts.append(extra_suffix.strip())
    return ", ".join(p for p in parts if p)


def build_music_prompt(theme: str) -> str:
    return (
        f"Ambient cinematic background music inspired by: {theme}. "
        "Atmospheric, evolving textures, subtle rhythm, orchestral and "
        "electronic elements, suitable as film underscore, no vocals."
    )
