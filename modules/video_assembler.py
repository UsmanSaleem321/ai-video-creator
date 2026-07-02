"""Final video assembly using MoviePy.

Combines per-scene images, narration audio, background music, subtitles,
and crossfade transitions into a single 1080p H.264 MP4.
"""

from pathlib import Path
from typing import List, Optional

from tqdm.auto import tqdm

from config import Config
from utils.helpers import clear_gpu_memory, timer

try:
    from moviepy.editor import (
        ImageClip,
        AudioFileClip,
        CompositeVideoClip,
        CompositeAudioClip,
        TextClip,
        concatenate_videoclips,
    )
    from moviepy.audio.fx.all import volumex, audio_loop
    from moviepy.video.fx.all import resize as mp_resize, crop as mp_crop
except ImportError:
    ImageClip = AudioFileClip = CompositeVideoClip = CompositeAudioClip = None
    TextClip = concatenate_videoclips = None
    volumex = audio_loop = mp_resize = mp_crop = None


class VideoAssemblyError(RuntimeError):
    pass


class VideoAssembler:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = self.config.get_logger("video_assembler")
        self._subtitles_supported = True

    # ------------------------------------------------------------- helpers
    def _fit_to_frame(self, clip):
        """Scale + center-crop an image clip to exactly fill the target resolution."""
        target_w, target_h = self.config.video_width, self.config.video_height
        target_ratio = target_w / target_h
        clip_ratio = clip.w / clip.h

        if clip_ratio > target_ratio:
            clip = clip.fx(mp_resize, height=target_h)
        else:
            clip = clip.fx(mp_resize, width=target_w)

        clip = clip.fx(
            mp_crop,
            x_center=clip.w / 2,
            y_center=clip.h / 2,
            width=target_w,
            height=target_h,
        )
        return clip

    def _make_subtitle_clip(self, text: str, duration: float):
        if not self._subtitles_supported:
            return None
        try:
            txt_clip = TextClip(
                text,
                fontsize=self.config.subtitle_fontsize,
                color=self.config.subtitle_color,
                stroke_color=self.config.subtitle_stroke_color,
                stroke_width=self.config.subtitle_stroke_width,
                font=self.config.subtitle_font,
                method="caption",
                size=(int(self.config.video_width * 0.85), None),
                align="center",
            )
            return (
                txt_clip.set_duration(duration)
                .set_position(("center", 0.85), relative=True)
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "Subtitles disabled: TextClip failed (likely missing ImageMagick): %s", exc
            )
            self._subtitles_supported = False
            return None

    def _build_scene_clip(self, image_path: Path, narration_path: Optional[Path], text: str):
        narration_audio = AudioFileClip(str(narration_path)) if narration_path else None
        min_duration = self.config.scene_duration_seconds
        duration = max(min_duration, narration_audio.duration + 0.5) if narration_audio else min_duration

        img_clip = ImageClip(str(image_path)).set_duration(duration)
        img_clip = self._fit_to_frame(img_clip)

        layers = [img_clip]
        subtitle_clip = self._make_subtitle_clip(text, duration)
        if subtitle_clip is not None:
            layers.append(subtitle_clip)

        scene_clip = CompositeVideoClip(layers, size=(self.config.video_width, self.config.video_height))
        scene_clip = scene_clip.set_duration(duration)
        if narration_audio is not None:
            narration_audio = narration_audio.fx(volumex, self.config.narration_volume)
            scene_clip = scene_clip.set_audio(narration_audio)
        return scene_clip

    # ---------------------------------------------------- cinematic mode
    def _build_scene_clip_from_video(
        self,
        clip_path: Path,
        narration_path: Optional[Path],
        text: str,
        zoom_amount: float = 0.0,
    ):
        """Video-clip analog of _build_scene_clip: loads a short generated
        clip, loops/holds it to match narration length, fits to frame,
        optionally bakes in a continuous zoom (used for scene transitions so
        they read as real camera motion rather than a flat dissolve), then
        overlays the subtitle and attaches narration audio.
        """
        from moviepy.editor import VideoFileClip
        from moviepy.video.fx.all import loop as mp_loop

        video_clip = VideoFileClip(str(clip_path))
        narration_audio = AudioFileClip(str(narration_path)) if narration_path else None
        min_duration = max(self.config.scene_duration_seconds, video_clip.duration)
        duration = max(min_duration, narration_audio.duration + 0.5) if narration_audio else min_duration

        if duration > video_clip.duration:
            video_clip = video_clip.fx(mp_loop, duration=duration)
        else:
            video_clip = video_clip.subclip(0, duration)

        video_clip = self._fit_to_frame(video_clip)

        if zoom_amount:
            # resize grows the frame from its top-left corner; the
            # surrounding CompositeVideoClip's fixed canvas size clips the
            # overflow automatically, so no separate crop step is needed.
            video_clip = video_clip.fx(mp_resize, lambda t: 1.0 + zoom_amount * (t / duration))

        layers = [video_clip]
        subtitle_clip = self._make_subtitle_clip(text, duration)
        if subtitle_clip is not None:
            layers.append(subtitle_clip)

        scene_clip = CompositeVideoClip(layers, size=(self.config.video_width, self.config.video_height))
        scene_clip = scene_clip.set_duration(duration)
        if narration_audio is not None:
            narration_audio = narration_audio.fx(volumex, self.config.narration_volume)
            scene_clip = scene_clip.set_audio(narration_audio)
        return scene_clip

    def assemble_cinematic(
        self,
        scenes: List,
        clip_paths: List[Path],
        narration_paths: List[Path],
        music_path: Optional[Path],
        output_name: str = "final_video.mp4",
    ) -> Path:
        """Cinematic-mode analog of assemble(): concatenates real per-scene
        video clips (instead of static ImageClips) with a zoom-crossfade
        transition, same subtitle/narration/music mixing as the slideshow
        path.
        """
        if ImageClip is None:
            raise VideoAssemblyError("moviepy is not installed")
        if not (len(scenes) == len(clip_paths) == len(narration_paths)):
            raise VideoAssemblyError(
                f"Mismatched counts: scenes={len(scenes)}, clips={len(clip_paths)}, "
                f"narration={len(narration_paths)}"
            )

        transition = self.config.transition_duration_seconds
        zoom_amount = self.config.cinematic_transition_zoom_amount
        scene_clips = []
        with timer("building per-scene video clips", self.logger):
            for i, (scene, clip_path, narration_path) in enumerate(
                tqdm(list(zip(scenes, clip_paths, narration_paths)), desc="Assembling cinematic scenes")
            ):
                clip = self._build_scene_clip_from_video(
                    clip_path, narration_path, scene.narration,
                    zoom_amount=zoom_amount if i > 0 else 0.0,
                )
                if i > 0:
                    clip = clip.crossfadein(transition)
                scene_clips.append(clip)

        with timer("concatenating scenes with zoom-crossfade", self.logger):
            final_video = concatenate_videoclips(
                scene_clips, padding=-transition, method="compose"
            )

        final_video = self._mix_audio(final_video, music_path)

        output_path = self.config.output_dir / output_name
        with timer("exporting final cinematic MP4", self.logger):
            final_video.write_videofile(
                str(output_path),
                fps=self.config.fps,
                codec=self.config.video_codec,
                audio_codec=self.config.audio_codec,
                preset="medium",
                threads=4,
                temp_audiofile=str(self.config.cache_dir / "temp-audio-cinematic.m4a"),
                remove_temp=True,
            )

        for clip in scene_clips:
            clip.close()
        final_video.close()
        clear_gpu_memory()
        return output_path

    # --------------------------------------------------------------- build
    def assemble(
        self,
        scenes: List,
        image_paths: List[Path],
        narration_paths: List[Path],
        music_path: Optional[Path],
        output_name: str = "final_video.mp4",
    ) -> Path:
        if ImageClip is None:
            raise VideoAssemblyError("moviepy is not installed")

        if not (len(scenes) == len(image_paths) == len(narration_paths)):
            raise VideoAssemblyError(
                f"Mismatched counts: scenes={len(scenes)}, images={len(image_paths)}, "
                f"narration={len(narration_paths)}"
            )

        transition = self.config.transition_duration_seconds
        scene_clips = []
        with timer("building per-scene clips", self.logger):
            for i, (scene, image_path, narration_path) in enumerate(
                tqdm(list(zip(scenes, image_paths, narration_paths)), desc="Assembling scenes")
            ):
                clip = self._build_scene_clip(image_path, narration_path, scene.narration)
                if i > 0:
                    clip = clip.crossfadein(transition)
                scene_clips.append(clip)

        with timer("concatenating scenes with crossfade", self.logger):
            final_video = concatenate_videoclips(
                scene_clips, padding=-transition, method="compose"
            )

        final_video = self._mix_audio(final_video, music_path)

        output_path = self.config.output_dir / output_name
        with timer("exporting final MP4", self.logger):
            final_video.write_videofile(
                str(output_path),
                fps=self.config.fps,
                codec=self.config.video_codec,
                audio_codec=self.config.audio_codec,
                preset="medium",
                threads=4,
                temp_audiofile=str(self.config.cache_dir / "temp-audio.m4a"),
                remove_temp=True,
            )

        for clip in scene_clips:
            clip.close()
        final_video.close()
        clear_gpu_memory()
        return output_path

    def _mix_audio(self, final_video, music_path: Optional[Path]):
        if not music_path or not Path(music_path).exists():
            self.logger.warning("No background music provided; exporting narration-only audio")
            return final_video

        narration_track = final_video.audio
        music_clip = AudioFileClip(str(music_path))
        music_clip = music_clip.fx(audio_loop, duration=final_video.duration)
        music_clip = music_clip.fx(volumex, self.config.music_volume)

        if narration_track is not None:
            combined_audio = CompositeAudioClip([narration_track, music_clip])
        else:
            combined_audio = music_clip

        return final_video.set_audio(combined_audio)


def assemble_final_video(
    scenes: List,
    image_paths: List[Path],
    narration_paths: List[Path],
    music_path: Optional[Path],
    output_name: str = "final_video.mp4",
    config: Optional[Config] = None,
) -> Path:
    assembler = VideoAssembler(config)
    return assembler.assemble(scenes, image_paths, narration_paths, music_path, output_name)
