#!/usr/bin/env python
"""AI Video Creator - end-to-end pipeline entry point.

Usage (CLI):
    python main.py --theme "A cyberpunk city with neon lights and flying cars at night" \
        --output cyberpunk_city.mp4

Usage (as a library, e.g. from the Colab notebook):
    from main import create_video
    video_path = create_video(
        theme="A cyberpunk city with neon lights and flying cars at night",
        output_name="cyberpunk_city.mp4",
    )
"""

import argparse
import sys
import traceback
from pathlib import Path
from typing import Optional, Tuple

from config import Config
from utils.helpers import clear_gpu_memory, timer, slugify
from modules.story_generator import StoryGenerator
from modules.image_generator import ImageGenerator
from modules.audio_generator import NarrationGenerator, MusicGenerator
from modules.video_assembler import VideoAssembler
from modules.video_generator import VideoGenerator
from modules.character_consistency import CharacterReferenceStore
from modules.motion_controller import MotionController


def create_video(
    theme: str,
    output_name: Optional[str] = None,
    style: str = "cinematic",
    quality: str = "balanced",
    num_scenes: Optional[int] = None,
) -> Path:
    """Run the full pipeline: story -> images -> narration -> music -> video.

    Each stage loads its own model(s), does its work, then unloads before the
    next stage starts. This keeps peak GPU memory usage to "one model at a
    time" so the pipeline fits on a single Colab GPU (e.g. T4, 15GB).
    """
    config = Config()
    config.quality_preset = quality
    config._apply_quality_preset()
    logger = config.get_logger("main")

    if not output_name:
        output_name = f"{slugify(theme)}.mp4"
    if not output_name.endswith(".mp4"):
        output_name += ".mp4"

    logger.info("=" * 70)
    logger.info("AI VIDEO CREATOR")
    logger.info("Theme: %s", theme)
    logger.info("Device: %s | FP16: %s | VRAM: %.1fGB", config.device, config.use_fp16, config.gpu_vram_gb)
    logger.info("=" * 70)

    # ---------------------------------------------------------- 1. story
    story_gen = StoryGenerator(config)
    try:
        with timer("STEP 1/5: Story generation", logger):
            scenes = story_gen.generate_story(theme, num_scenes=num_scenes)
        logger.info("Generated %d scenes", len(scenes))
    except Exception:
        logger.error("Story generation failed:\n%s", traceback.format_exc())
        raise
    finally:
        story_gen.unload()
        clear_gpu_memory()

    # ---------------------------------------------------------- 2. images
    image_gen = ImageGenerator(config)
    try:
        with timer("STEP 2/5: Image generation", logger):
            image_paths = image_gen.generate_images_for_scenes(scenes, style=style)
        logger.info("Generated %d images", len(image_paths))
    except Exception:
        logger.error("Image generation failed:\n%s", traceback.format_exc())
        raise
    finally:
        image_gen.unload()
        clear_gpu_memory()

    # ------------------------------------------------------- 3. narration
    narration_gen = NarrationGenerator(config)
    try:
        with timer("STEP 3/5: Narration generation", logger):
            narration_paths = narration_gen.generate_narration_for_scenes(scenes)
        logger.info("Generated %d narration clips", len(narration_paths))
    except Exception:
        logger.error("Narration generation failed:\n%s", traceback.format_exc())
        raise
    finally:
        narration_gen.unload()
        clear_gpu_memory()

    # ----------------------------------------------------------- 4. music
    music_gen = MusicGenerator(config)
    try:
        with timer("STEP 4/5: Background music generation", logger):
            total_duration = len(scenes) * config.scene_duration_seconds
            music_path = music_gen.generate_music(theme, duration=min(60.0, max(30.0, total_duration)))
        logger.info("Generated background music: %s", music_path)
    except Exception:
        logger.error("Music generation failed, continuing without music:\n%s", traceback.format_exc())
        music_path = None
    finally:
        music_gen.unload()
        clear_gpu_memory()

    # ----------------------------------------------------------- 5. video
    assembler = VideoAssembler(config)
    with timer("STEP 5/5: Video assembly", logger):
        video_path = assembler.assemble(
            scenes=scenes,
            image_paths=image_paths,
            narration_paths=narration_paths,
            music_path=music_path,
            output_name=output_name,
        )

    logger.info("=" * 70)
    logger.info("DONE. Final video saved to: %s", video_path)
    logger.info("=" * 70)
    return video_path


def _resolve_scene_motion(motion_controller: MotionController, scene) -> Tuple[Optional[dict], Optional[str], Optional[object], str]:
    """Translate a Scene's LLM-tagged `motion` name (see utils/prompts.py's
    MOTION: field and story_generator.StoryGenerator._normalize_motion) into
    the knobs modules/video_generator.py and modules/image_generator.py
    actually take: an SVD motion_hint (intensity), a post-generation
    camera_motion_type OR action_plan (direction/content), and a prompt
    suffix (action presets only, so the keyframe itself depicts the action
    rather than just having camera motion added on top of an unrelated
    image).

    Returns (motion_hint, camera_motion_type, action_plan, prompt_suffix).
    """
    motion_name = getattr(scene, "motion", "none") or "none"
    if motion_name == "none":
        return None, None, None, ""
    if motion_name in motion_controller.ACTION_SEQUENCES:
        plan = motion_controller.create_action_sequence(scene.visual, action_type=motion_name)
        return plan.motion_hint, None, plan, plan.prompt_suffix
    if motion_name in motion_controller.CAMERA_MOTIONS or motion_name in motion_controller.SUBJECT_MOTIONS:
        return motion_controller.resolve_motion_hint(motion_name), motion_name, None, ""
    return None, None, None, ""


def create_cinematic_video(
    theme: str,
    output_name: Optional[str] = None,
    style: str = "cinematic",
    quality: str = "balanced",
    num_scenes: Optional[int] = None,
) -> Path:
    """Cinematic pipeline: story+characters -> character refs -> per-scene
    keyframes (IP-Adapter conditioned) -> per-scene video clips (SVD) ->
    narration -> music -> real-clip assembly.

    Mirrors create_video()'s "load, use, unload" per-stage discipline.
    Keyframe generation and SVD generation are two separate load/unload
    stages (all keyframes first, then all clips) so SD1.5+IP-Adapter and SVD
    are never resident on the GPU at the same time.
    """
    config = Config()
    config.quality_preset = quality
    config._apply_quality_preset()
    config.pipeline_mode = "cinematic"
    logger = config.get_logger("main")

    if not output_name:
        output_name = f"{slugify(theme)}_cinematic.mp4"
    if not output_name.endswith(".mp4"):
        output_name += ".mp4"

    logger.info("=" * 70)
    logger.info("AI VIDEO CREATOR (cinematic mode)")
    logger.info("Theme: %s", theme)
    logger.info("Device: %s | FP16: %s | VRAM: %.1fGB", config.device, config.use_fp16, config.gpu_vram_gb)
    logger.info("=" * 70)

    # ------------------------------------------------- 1. story + characters
    story_gen = StoryGenerator(config)
    try:
        with timer("STEP 1/6: Story + character generation", logger):
            scenes, characters = story_gen.generate_story_with_characters(theme, num_scenes=num_scenes)
        logger.info("Generated %d scenes, %d characters", len(scenes), len(characters))
    except Exception:
        logger.error("Story generation failed:\n%s", traceback.format_exc())
        raise
    finally:
        story_gen.unload()
        clear_gpu_memory()

    # ---------------------------------------------------- 2a. motion planning
    # Resolve each scene's LLM-tagged MOTION preset (see utils/prompts.py)
    # into: an SVD intensity hint, a post-generation camera/subject motion
    # name or action plan, and (for action presets only) a prompt suffix so
    # the keyframe itself depicts the action.
    motion_controller = MotionController(config)
    motion_hints, camera_motion_types, action_plans, motion_prompt_suffixes = [], [], [], []
    for scene in scenes:
        hint, cam_type, plan, suffix = _resolve_scene_motion(motion_controller, scene)
        motion_hints.append(hint)
        camera_motion_types.append(cam_type)
        action_plans.append(plan)
        motion_prompt_suffixes.append(suffix)
    logger.info(
        "Scene motion presets: %s",
        ", ".join(f"{s.index}={s.motion}" for s in scenes),
    )

    # --------------------------------------- 2b. character refs + keyframes
    image_gen = ImageGenerator(config)
    character_store = CharacterReferenceStore(config)
    try:
        with timer("STEP 2/6: Character references + keyframes", logger):
            image_gen.load_model()
            if characters:
                character_store.generate_references(image_gen.pipe, characters)
                character_store.attach_ip_adapter(image_gen.pipe)
            keyframe_paths = image_gen.generate_keyframes_for_scenes(
                scenes,
                style=style,
                character_store=character_store if characters else None,
                extra_prompt_suffixes=motion_prompt_suffixes,
            )
        logger.info("Generated %d keyframes", len(keyframe_paths))
    except Exception:
        logger.error("Keyframe generation failed:\n%s", traceback.format_exc())
        raise
    finally:
        character_store.unload()
        image_gen.unload()
        clear_gpu_memory()

    # ----------------------------------------------------- 3. video clips
    video_gen = VideoGenerator(config)
    try:
        with timer("STEP 3/6: Scene video generation (Stable Video Diffusion)", logger):
            clip_paths = video_gen.generate_clips_for_scenes(
                scenes,
                keyframe_paths,
                motion_hints=motion_hints,
                camera_motion_types=camera_motion_types,
                action_plans=action_plans,
                motion_controller=motion_controller,
            )
        logger.info("Generated %d video clips", len(clip_paths))
    except Exception:
        logger.error("Video generation failed:\n%s", traceback.format_exc())
        raise
    finally:
        video_gen.unload()
        clear_gpu_memory()

    # ------------------------------------------------------- 4. narration
    narration_gen = NarrationGenerator(config)
    try:
        with timer("STEP 4/6: Narration generation", logger):
            narration_paths = narration_gen.generate_narration_for_scenes(scenes)
        logger.info("Generated %d narration clips", len(narration_paths))
    except Exception:
        logger.error("Narration generation failed:\n%s", traceback.format_exc())
        raise
    finally:
        narration_gen.unload()
        clear_gpu_memory()

    # ----------------------------------------------------------- 5. music
    music_gen = MusicGenerator(config)
    try:
        with timer("STEP 5/6: Background music generation", logger):
            total_duration = len(scenes) * config.scene_duration_seconds
            music_path = music_gen.generate_music(theme, duration=min(60.0, max(30.0, total_duration)))
        logger.info("Generated background music: %s", music_path)
    except Exception:
        logger.error("Music generation failed, continuing without music:\n%s", traceback.format_exc())
        music_path = None
    finally:
        music_gen.unload()
        clear_gpu_memory()

    # ----------------------------------------------------------- 6. assemble
    assembler = VideoAssembler(config)
    with timer("STEP 6/6: Cinematic video assembly", logger):
        video_path = assembler.assemble_cinematic(
            scenes=scenes,
            clip_paths=clip_paths,
            narration_paths=narration_paths,
            music_path=music_path,
            output_name=output_name,
        )

    logger.info("=" * 70)
    logger.info("DONE. Final cinematic video saved to: %s", video_path)
    logger.info("=" * 70)
    return video_path


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate a cinematic AI video from a text theme.")
    parser.add_argument("--theme", required=True, help="Text theme/prompt to build the video around")
    parser.add_argument("--output", default=None, help="Output filename (default: derived from theme)")
    parser.add_argument(
        "--style",
        default="cinematic",
        choices=["cinematic", "concept_art", "photorealistic", "anime", "noir"],
        help="Visual style for image generation",
    )
    parser.add_argument(
        "--quality",
        default="balanced",
        choices=["draft", "balanced", "high"],
        help="Speed/quality trade-off preset",
    )
    parser.add_argument("--num-scenes", type=int, default=None, help="Override number of scenes (5-7 recommended)")
    parser.add_argument(
        "--mode",
        default="slideshow",
        choices=["slideshow", "cinematic"],
        help=(
            "slideshow: static images per scene (original pipeline). "
            "cinematic: real short video clips per scene via SD1.5+IP-Adapter "
            "character-consistent keyframes animated with Stable Video Diffusion."
        ),
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    try:
        pipeline_fn = create_cinematic_video if args.mode == "cinematic" else create_video
        video_path = pipeline_fn(
            theme=args.theme,
            output_name=args.output,
            style=args.style,
            quality=args.quality,
            num_scenes=args.num_scenes,
        )
        print(f"\nVideo created successfully: {video_path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"\nPipeline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
