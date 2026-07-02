"""Cinematic camera motion, subject motion, and action-sequence presets.

Read this before wiring anything up — it explains what's genuinely
controllable here versus what's an approximation, so nothing downstream
gets built on a false assumption:

The pipeline's only video-generation backend today is Stable Video
Diffusion (modules/video_generator.py). SVD takes a single scalar
`motion_bucket_id` (+ `noise_aug_strength`) as its *only* motion control —
it has no concept of direction; it cannot be told "pan left" vs "pan
right", or "zoom in" vs "zoom out". So this module does two genuinely
different things, and keeps them clearly separate:

  1. Pre-generation *intensity* hints (`resolve_motion_hint`): maps a named
     preset to an SVD `motion_bucket_id` / `noise_aug_strength` pair. This
     only controls how much the model animates the frame, never which
     direction.
  2. Post-generation *frame-space* camera transforms (`add_camera_motion`,
     `add_subject_motion`, `apply_action_sequence`): real, deterministic
     pan/zoom/tilt/rotate/shake applied directly to already-rendered frames
     via a classic Ken-Burns-style crop-and-resize (or PIL rotate) per
     frame. Directional control — left vs right, up vs down, in vs out —
     comes from here, not from the diffusion model. These functions work on
     ANY frame list (SVD output, or a single static keyframe repeated N
     times), so they're equally usable to add camera motion on top of the
     slideshow pipeline's still images.

"Subject motion" (walk/run/fly/jump/rotate) reuses the same whole-frame
transform primitives as camera motion. There is no character/object
segmentation in this pipeline, so these functions move the *whole frame* in
a way that reads as that kind of motion (e.g. walk_forward = a slow forward
dolly plus a subtle vertical bob) — they do not animate an isolated
character independently of its background. True isolated-subject motion
would need a pose- or segmentation-driven video model (e.g. ControlNet +
AnimateDiff), which is out of scope for this module.

"Action sequences" (fight/chase/explosion/fly_through) can't be *generated*
by this module either — the actual fighting/chasing/exploding content has
to come from the diffusion model conditioned on a good prompt. What this
module contributes for an action sequence is: (a) a prompt suffix to steer
the keyframe/story prompt toward that action, (b) an SVD motion hint tuned
for that action's intensity, and (c) an ordered sequence of the
camera-motion primitives above (e.g. chase = tracking_shot -> zoom_in,
explosion = zoom_out + camera shake + a flash) applied across the clip's
frames.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image

from config import Config
from utils.helpers import timer

FrameList = List[Image.Image]


# --------------------------------------------------------------------------
# Core transform primitives
# --------------------------------------------------------------------------
def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _ease_in_out(t: float) -> float:
    """Smoothstep easing so motion accelerates/decelerates instead of moving
    at a constant speed — reads as more deliberately "cinematic"."""
    return t * t * (3 - 2 * t)


def _crop_zoom_pan(image: Image.Image, zoom: float, pan_x: float, pan_y: float) -> Image.Image:
    """Crop a sub-rectangle of `image` (a Ken-Burns-style camera move) and
    resize it back to the original frame size.

    zoom: >1.0 crops a smaller box ("zoomed in"); 1.0 = full frame, no room
        to pan.
    pan_x, pan_y: crop-box center offset as a fraction (-1.0..1.0) of
        whatever slack the zoomed-in crop box leaves to move within the
        original frame.
    """
    width, height = image.size
    zoom = max(zoom, 1.0)
    crop_w = width / zoom
    crop_h = height / zoom

    max_offset_x = (width - crop_w) / 2
    max_offset_y = (height - crop_h) / 2

    center_x = width / 2 + _clamp(pan_x, -1.0, 1.0) * max_offset_x
    center_y = height / 2 + _clamp(pan_y, -1.0, 1.0) * max_offset_y

    left = _clamp(center_x - crop_w / 2, 0, width - crop_w)
    top = _clamp(center_y - crop_h / 2, 0, height - crop_h)

    box = (left, top, left + crop_w, top + crop_h)
    return image.resize((width, height), resample=Image.BICUBIC, box=box)


def _camera_shake(image: Image.Image, magnitude: float, rng: random.Random) -> Image.Image:
    dx = rng.uniform(-magnitude, magnitude)
    dy = rng.uniform(-magnitude, magnitude)
    return _crop_zoom_pan(image, 1.1, dx, dy)


def _flash(image: Image.Image, strength: float) -> Image.Image:
    """Blend the frame toward white — used for an explosion's flash frames."""
    if strength <= 0:
        return image
    rgb = image.convert("RGB")
    white = Image.new("RGB", rgb.size, (255, 255, 255))
    return Image.blend(rgb, white, min(strength, 1.0))


def _apply_frame_transform(
    frames: FrameList, transform_fn: Callable[[Image.Image, float], Image.Image]
) -> FrameList:
    """Apply transform_fn(frame, t) to every frame, where t in [0, 1] is
    that frame's position within the clip (0 = first frame, 1 = last)."""
    n = len(frames)
    if n == 0:
        return frames
    if n == 1:
        return [transform_fn(frames[0], 0.0)]
    return [transform_fn(frame, i / (n - 1)) for i, frame in enumerate(frames)]


# --------------------------------------------------------------------------
# Camera motion presets
# --------------------------------------------------------------------------
_PAN_ZOOM = 1.15  # baseline zoom-in that gives pan/crane/tracking room to move


def pan_left(frames: FrameList, intensity: float = 0.8) -> FrameList:
    """Camera sweeps from right to left across the frame."""
    return _apply_frame_transform(
        frames, lambda f, t: _crop_zoom_pan(f, _PAN_ZOOM, _lerp(intensity, -intensity, t), 0.0)
    )


def pan_right(frames: FrameList, intensity: float = 0.8) -> FrameList:
    """Camera sweeps from left to right across the frame."""
    return _apply_frame_transform(
        frames, lambda f, t: _crop_zoom_pan(f, _PAN_ZOOM, _lerp(-intensity, intensity, t), 0.0)
    )


def zoom_in(frames: FrameList, intensity: float = 0.22) -> FrameList:
    """Camera gradually zooms in toward the frame center."""
    return _apply_frame_transform(frames, lambda f, t: _crop_zoom_pan(f, 1.0 + intensity * t, 0.0, 0.0))


def zoom_out(frames: FrameList, intensity: float = 0.22) -> FrameList:
    """Camera starts zoomed in and gradually pulls back to the full frame."""
    return _apply_frame_transform(frames, lambda f, t: _crop_zoom_pan(f, 1.0 + intensity * (1 - t), 0.0, 0.0))


def dolly_zoom(frames: FrameList, intensity: float = 0.25) -> FrameList:
    """Approximates a "vertigo" dolly-zoom: zooms in on the frame center
    while very slightly stretching it, faking the perspective shift a real
    dolly zoom gets from an actual depth/focal-length change (not possible
    from a single flat 2D frame)."""

    def _transform(f: Image.Image, t: float) -> Image.Image:
        zoomed = _crop_zoom_pan(f, 1.0 + intensity * t, 0.0, 0.0)
        stretch = 1.0 + 0.04 * t
        w, h = zoomed.size
        stretched = zoomed.resize((max(1, int(w * stretch)), h), resample=Image.BICUBIC)
        left = (stretched.width - w) // 2
        return stretched.crop((left, 0, left + w, h))

    return _apply_frame_transform(frames, _transform)


def orbit(frames: FrameList, intensity: float = 0.5) -> FrameList:
    """Approximates an orbiting camera via a horizontal sweep out-and-back
    combined with a gentle zoom pulse (a true 3D orbit needs depth
    information a single 2D frame doesn't have)."""

    def _transform(f: Image.Image, t: float) -> Image.Image:
        pan_x = math.sin(t * math.pi) * intensity
        zoom = 1.1 + 0.05 * math.sin(t * 2 * math.pi)
        return _crop_zoom_pan(f, zoom, pan_x, 0.0)

    return _apply_frame_transform(frames, _transform)


def crane_up(frames: FrameList, intensity: float = 0.8) -> FrameList:
    """Camera rises, sweeping from the bottom of the frame to the top."""
    return _apply_frame_transform(
        frames, lambda f, t: _crop_zoom_pan(f, _PAN_ZOOM, 0.0, _lerp(intensity, -intensity, t))
    )


def crane_down(frames: FrameList, intensity: float = 0.8) -> FrameList:
    """Camera descends, sweeping from the top of the frame to the bottom."""
    return _apply_frame_transform(
        frames, lambda f, t: _crop_zoom_pan(f, _PAN_ZOOM, 0.0, _lerp(-intensity, intensity, t))
    )


def tracking_shot(frames: FrameList, direction: str = "right", intensity: float = 0.7) -> FrameList:
    """Camera tracks smoothly in the given direction: 'left', 'right', 'up', or 'down'."""
    dx, dy = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}.get(direction, (1, 0))

    def _transform(f: Image.Image, t: float) -> Image.Image:
        pan = _lerp(-intensity, intensity, t)
        return _crop_zoom_pan(f, _PAN_ZOOM, dx * pan, dy * pan)

    return _apply_frame_transform(frames, _transform)


def dynamic_tracking(frames: FrameList, intensity: float = 0.85) -> FrameList:
    """Like tracking_shot but with eased (accelerate-then-decelerate)
    horizontal motion and a slight forward push, for a more energetic
    action-camera feel."""
    zoom_range = 0.15

    def _transform(f: Image.Image, t: float) -> Image.Image:
        eased = _ease_in_out(t)
        pan = _lerp(-intensity, intensity, eased)
        zoom = 1.1 + zoom_range * eased
        return _crop_zoom_pan(f, zoom, pan, 0.0)

    return _apply_frame_transform(frames, _transform)


# --------------------------------------------------------------------------
# Subject motion presets
#
# See module docstring: these move the whole frame in a way that *reads* as
# the named motion; they do not animate an isolated character.
# --------------------------------------------------------------------------
def walk_forward(frames: FrameList, intensity: float = 0.12) -> FrameList:
    """Slow forward dolly plus a subtle vertical bob, approximating the feel
    of forward walking motion."""

    def _transform(f: Image.Image, t: float) -> Image.Image:
        zoom = 1.0 + intensity * t
        bob = 0.02 * math.sin(t * math.pi * 6)
        return _crop_zoom_pan(f, zoom, 0.0, bob)

    return _apply_frame_transform(frames, _transform)


def run_forward(frames: FrameList, intensity: float = 0.22) -> FrameList:
    """Faster forward dolly with a quicker, larger vertical bob."""

    def _transform(f: Image.Image, t: float) -> Image.Image:
        zoom = 1.0 + intensity * t
        bob = 0.035 * math.sin(t * math.pi * 10)
        return _crop_zoom_pan(f, zoom, 0.0, bob)

    return _apply_frame_transform(frames, _transform)


def fly_up(frames: FrameList, intensity: float = 0.8) -> FrameList:
    """Frame sweeps upward with a slight zoom-out, suggesting gained altitude."""
    return _apply_frame_transform(
        frames, lambda f, t: _crop_zoom_pan(f, _PAN_ZOOM - 0.05 * t, 0.0, _lerp(intensity, -intensity, t))
    )


def fly_down(frames: FrameList, intensity: float = 0.8) -> FrameList:
    """Frame sweeps downward, suggesting a descent."""
    return _apply_frame_transform(
        frames, lambda f, t: _crop_zoom_pan(f, _PAN_ZOOM, 0.0, _lerp(-intensity, intensity, t))
    )


def jump(frames: FrameList, intensity: float = 0.5) -> FrameList:
    """A single parabolic vertical arc (up then back down), like a jump,
    rather than a repeating bob."""

    def _transform(f: Image.Image, t: float) -> Image.Image:
        arc = 4 * t * (1 - t)  # 0 -> 1 -> 0, peaking at t=0.5
        zoom = 1.05 + 0.05 * arc
        return _crop_zoom_pan(f, zoom, 0.0, -intensity * arc)

    return _apply_frame_transform(frames, _transform)


def rotate(frames: FrameList, degrees: float = 15.0, direction: str = "cw") -> FrameList:
    """Incrementally rotates the frame across the clip (a subtle Dutch-angle
    style rotation, not a full spin — large angles crop away too much of a
    non-square source frame to look intentional; corners left by the
    rotation are cropped out via a small zoom-in)."""
    sign = 1 if direction == "cw" else -1

    def _transform(f: Image.Image, t: float) -> Image.Image:
        angle = sign * degrees * t
        rotated = f.rotate(-angle, resample=Image.BICUBIC, expand=False)
        return _crop_zoom_pan(rotated, 1.12, 0.0, 0.0)

    return _apply_frame_transform(frames, _transform)


# --------------------------------------------------------------------------
# Action sequences
# --------------------------------------------------------------------------
@dataclass
class ActionPlan:
    """What create_action_sequence() actually produces: not pixels, but a
    plan for two separate downstream steps — an SVD motion hint plus an
    ordered list of camera-motion segments — since generating the action's
    actual visual content requires the diffusion model, not this module.
    """

    action_type: str
    prompt_suffix: str
    motion_hint: dict
    camera_segments: List[Tuple[str, float]] = field(default_factory=list)  # (camera_motion_name, fraction)
    shake_intensity: float = 0.0
    flash_at: Optional[float] = None  # fraction of the clip where a flash peaks (e.g. explosions)


def fight_sequence(scene_prompt: str = "") -> ActionPlan:
    return ActionPlan(
        action_type="fight",
        prompt_suffix=(
            "intense hand-to-hand combat, dynamic mid-action pose, motion blur, "
            "dramatic action framing, sparks and impact"
        ),
        motion_hint={"motion_bucket_id": 180, "noise_aug_strength": 0.04},
        camera_segments=[("dynamic_tracking", 0.5), ("orbit", 0.5)],
        shake_intensity=0.35,
    )


def chase_sequence(scene_prompt: str = "") -> ActionPlan:
    return ActionPlan(
        action_type="chase",
        prompt_suffix=(
            "high speed chase, motion blur, dynamic low camera angle, "
            "urgent forward momentum, debris flying"
        ),
        motion_hint={"motion_bucket_id": 200, "noise_aug_strength": 0.035},
        camera_segments=[("tracking_shot", 0.6), ("zoom_in", 0.4)],
        shake_intensity=0.15,
    )


def explosion_sequence(scene_prompt: str = "") -> ActionPlan:
    return ActionPlan(
        action_type="explosion",
        prompt_suffix=(
            "massive explosion, fireball and debris, shockwave, dramatic "
            "lighting, cinematic destruction"
        ),
        motion_hint={"motion_bucket_id": 220, "noise_aug_strength": 0.05},
        camera_segments=[("zoom_out", 1.0)],
        shake_intensity=0.5,
        flash_at=0.08,
    )


def fly_through(scene_prompt: str = "") -> ActionPlan:
    return ActionPlan(
        action_type="fly_through",
        prompt_suffix=(
            "sweeping aerial drone shot flying through the scene, wide "
            "dynamic perspective, cinematic aerial movement"
        ),
        motion_hint={"motion_bucket_id": 160, "noise_aug_strength": 0.03},
        camera_segments=[("dolly_zoom", 1.0)],
        shake_intensity=0.05,
    )


# --------------------------------------------------------------------------
# Controller
# --------------------------------------------------------------------------
class MotionController:
    """Applies cinematic camera motion, subject motion, and action-sequence
    presets to a list of already-generated video frames (e.g. Stable Video
    Diffusion output from modules/video_generator.py), and resolves
    pre-generation SVD motion hints for the same presets.

    See the module docstring for exactly what's genuinely controllable at
    each of those two stages.
    """

    CAMERA_MOTIONS: Dict[str, Callable[..., FrameList]] = {
        "pan_left": pan_left,
        "pan_right": pan_right,
        "zoom_in": zoom_in,
        "zoom_out": zoom_out,
        "dolly_zoom": dolly_zoom,
        "orbit": orbit,
        "crane_up": crane_up,
        "crane_down": crane_down,
        "tracking_shot": tracking_shot,
        "dynamic_tracking": dynamic_tracking,
    }

    SUBJECT_MOTIONS: Dict[str, Callable[..., FrameList]] = {
        "walk_forward": walk_forward,
        "run_forward": run_forward,
        "fly_up": fly_up,
        "fly_down": fly_down,
        "jump": jump,
        "rotate": rotate,
    }

    ACTION_SEQUENCES: Dict[str, Callable[[str], ActionPlan]] = {
        "fight": fight_sequence,
        "chase": chase_sequence,
        "explosion": explosion_sequence,
        "fly_through": fly_through,
    }

    # Rough SVD motion_bucket_id / noise_aug_strength hints per named preset
    # (used by resolve_motion_hint() for the *pre-generation* SVD call).
    # These only vary intensity — SVD has no directional control (see module
    # docstring); actual left/right/up/down/in/out direction comes from the
    # post-generation frame transforms above.
    _INTENSITY_HINTS: Dict[str, dict] = {
        "pan_left": {"motion_bucket_id": 110, "noise_aug_strength": 0.02},
        "pan_right": {"motion_bucket_id": 110, "noise_aug_strength": 0.02},
        "zoom_in": {"motion_bucket_id": 90, "noise_aug_strength": 0.02},
        "zoom_out": {"motion_bucket_id": 90, "noise_aug_strength": 0.02},
        "dolly_zoom": {"motion_bucket_id": 130, "noise_aug_strength": 0.025},
        "orbit": {"motion_bucket_id": 120, "noise_aug_strength": 0.025},
        "crane_up": {"motion_bucket_id": 100, "noise_aug_strength": 0.02},
        "crane_down": {"motion_bucket_id": 100, "noise_aug_strength": 0.02},
        "tracking_shot": {"motion_bucket_id": 140, "noise_aug_strength": 0.03},
        "dynamic_tracking": {"motion_bucket_id": 170, "noise_aug_strength": 0.035},
        "walk_forward": {"motion_bucket_id": 130, "noise_aug_strength": 0.025},
        "run_forward": {"motion_bucket_id": 175, "noise_aug_strength": 0.035},
        "fly_up": {"motion_bucket_id": 150, "noise_aug_strength": 0.03},
        "fly_down": {"motion_bucket_id": 150, "noise_aug_strength": 0.03},
        "jump": {"motion_bucket_id": 160, "noise_aug_strength": 0.03},
        "rotate": {"motion_bucket_id": 120, "noise_aug_strength": 0.025},
    }

    def __init__(self, config: Optional[Config] = None, seed: Optional[int] = None):
        self.config = config or Config()
        self.logger = self.config.get_logger("motion_controller")
        self._rng = random.Random(seed)

    def apply_named_motion(self, video_frames: FrameList, motion_type: str, **kwargs) -> FrameList:
        """Generic per-scene dispatch entry point: applies `motion_type` as
        a camera-motion preset if it's one, otherwise as a subject-motion
        preset. Use this (rather than add_camera_motion/add_subject_motion
        directly) when the preset name comes from somewhere that doesn't
        distinguish the two categories, e.g. an LLM-tagged scene."""
        if motion_type in self.CAMERA_MOTIONS:
            return self.add_camera_motion(video_frames, motion_type, **kwargs)
        if motion_type in self.SUBJECT_MOTIONS:
            return self.add_subject_motion(video_frames, motion_type, **kwargs)
        raise ValueError(
            f"Unknown motion_type {motion_type!r}. Options: "
            f"{sorted(set(self.CAMERA_MOTIONS) | set(self.SUBJECT_MOTIONS))}"
        )

    def add_camera_motion(
        self, video_frames: FrameList, motion_type: str = "dolly_zoom", **kwargs
    ) -> FrameList:
        """Apply a named camera-motion preset to a list of frames. Extra
        kwargs (e.g. intensity=, direction=) are passed through to the
        underlying preset function."""
        fn = self.CAMERA_MOTIONS.get(motion_type)
        if fn is None:
            raise ValueError(f"Unknown camera motion_type {motion_type!r}. Options: {sorted(self.CAMERA_MOTIONS)}")
        with timer(f"camera motion: {motion_type}", self.logger):
            return fn(video_frames, **kwargs)

    def add_subject_motion(
        self, video_frames: FrameList, movement_type: str = "walk_forward", **kwargs
    ) -> FrameList:
        """Apply a named subject-motion preset to a list of frames (see
        module docstring for the whole-frame-approximation caveat)."""
        fn = self.SUBJECT_MOTIONS.get(movement_type)
        if fn is None:
            raise ValueError(
                f"Unknown movement_type {movement_type!r}. Options: {sorted(self.SUBJECT_MOTIONS)}"
            )
        with timer(f"subject motion: {movement_type}", self.logger):
            return fn(video_frames, **kwargs)

    def create_action_sequence(self, scene_prompt: str, action_type: str = "fight") -> ActionPlan:
        """Build an ActionPlan for the named action (prompt suffix + SVD
        motion hint + ordered camera-motion segments). Does not touch pixels
        — pass the plan to apply_action_sequence() once frames exist, and
        append `plan.prompt_suffix` to the scene's image/keyframe prompt."""
        fn = self.ACTION_SEQUENCES.get(action_type)
        if fn is None:
            raise ValueError(f"Unknown action_type {action_type!r}. Options: {sorted(self.ACTION_SEQUENCES)}")
        return fn(scene_prompt)

    def apply_action_sequence(self, video_frames: FrameList, action_plan: ActionPlan) -> FrameList:
        """Apply an ActionPlan's camera_segments (in order, split across the
        frame list proportionally to each segment's fraction) plus shake and
        flash effects, to already-generated frames."""
        n = len(video_frames)
        if n == 0:
            return video_frames

        segments = action_plan.camera_segments or [("dolly_zoom", 1.0)]
        total_fraction = sum(fraction for _, fraction in segments) or 1.0

        result: FrameList = []
        start = 0
        for motion_type, fraction in segments:
            if start >= n:
                break
            length = max(1, round(n * fraction / total_fraction))
            end = min(n, start + length)
            segment_frames = video_frames[start:end]
            if segment_frames:
                result.extend(self.add_camera_motion(segment_frames, motion_type))
            start = end
        if start < n:  # leftover frames from rounding
            result.extend(video_frames[start:])

        if action_plan.shake_intensity:
            result = [_camera_shake(f, action_plan.shake_intensity * 0.06, self._rng) for f in result]

        if action_plan.flash_at is not None and result:
            peak_idx = round(action_plan.flash_at * (len(result) - 1)) if len(result) > 1 else 0
            peak_idx = int(_clamp(peak_idx, 0, len(result) - 1))
            flash_window = max(1, round(len(result) * 0.06))
            for i in range(max(0, peak_idx - flash_window), min(len(result), peak_idx + flash_window + 1)):
                distance = abs(i - peak_idx) / flash_window
                result[i] = _flash(result[i], strength=max(0.0, 1.0 - distance))

        return result

    def resolve_motion_hint(self, motion_type: str) -> dict:
        """Pre-generation SVD motion hint for a named preset — matches the
        dict shape modules/video_generator.py's `motion_hint` parameter
        expects. Only controls intensity, not direction (see module
        docstring)."""
        return dict(
            self._INTENSITY_HINTS.get(
                motion_type,
                {"motion_bucket_id": self.config.svd_motion_bucket_id, "noise_aug_strength": self.config.svd_noise_aug_strength},
            )
        )


# Canonical vocabulary, exposed at module level so other modules (e.g. the
# story-generation prompt/parser, which tags each scene with one of these
# names) can reference a single source of truth instead of duplicating it.
CAMERA_MOTION_NAMES: Tuple[str, ...] = tuple(MotionController.CAMERA_MOTIONS.keys())
SUBJECT_MOTION_NAMES: Tuple[str, ...] = tuple(MotionController.SUBJECT_MOTIONS.keys())
ACTION_SEQUENCE_NAMES: Tuple[str, ...] = tuple(MotionController.ACTION_SEQUENCES.keys())
ALL_MOTION_PRESET_NAMES: Tuple[str, ...] = (
    ("none",) + CAMERA_MOTION_NAMES + SUBJECT_MOTION_NAMES + ACTION_SEQUENCE_NAMES
)
