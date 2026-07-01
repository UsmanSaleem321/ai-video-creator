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
from typing import Optional

from config import Config
from utils.helpers import clear_gpu_memory, timer, slugify
from modules.story_generator import StoryGenerator
from modules.image_generator import ImageGenerator
from modules.audio_generator import NarrationGenerator, MusicGenerator
from modules.video_assembler import VideoAssembler


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
    return parser.parse_args()


def main():
    args = _parse_args()
    try:
        video_path = create_video(
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
