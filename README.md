# AI Video Creator

Turn a one-line text theme into a narrated, scored, cinematic MP4 — fully
automated, designed to run in Google Colab.

**Pipeline:** theme → story (Mistral-7B) → scene images (Stable Diffusion) →
narration (Bark) → background music (Stable Audio Open) → assembled video
(MoviePy, crossfades + subtitles, 1080p H.264).

## Quick start (Colab)

1. Open `notebooks/AI_VIDEO_CREATOR.ipynb` in Google Colab.
2. Select a GPU runtime (Runtime → Change runtime type → GPU, ideally a T4 or better).
3. Run the cells top to bottom. The notebook mounts Drive, installs
   dependencies, clones this repo (or uses the local copy), and runs the
   pipeline with progress bars.
4. The final video is saved to `output/` and copied to your Google Drive.

## Quick start (local / any machine with a CUDA GPU)

```bash
git clone <this-repo-url> ai-video-creator
cd ai-video-creator
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in HF_TOKEN
python main.py --theme "A cyberpunk city with neon lights and flying cars at night" --output cyberpunk_city.mp4
```

## Usage as a library

```python
from main import create_video

video_path = create_video(
    theme="A cyberpunk city with neon lights and flying cars at night",
    output_name="cyberpunk_city.mp4",
    style="cinematic",     # cinematic | concept_art | photorealistic | anime | noir
    quality="balanced",    # draft | balanced | high
)
```

## Project layout

```
ai-video-creator/
├── main.py                  # pipeline orchestration + CLI
├── config.py                 # Config dataclass: paths, device, model names, video settings
├── modules/
│   ├── story_generator.py    # Mistral-7B story generation + scene parsing
│   ├── image_generator.py    # Stable Diffusion 1.5 scene images
│   ├── audio_generator.py    # Bark narration + Stable Audio Open music
│   └── video_assembler.py    # MoviePy: crossfades, subtitles, mixing, export
├── utils/
│   ├── helpers.py            # GPU memory mgmt, retry, timing, JSON I/O
│   └── prompts.py            # prompt templates for story/image/music
├── notebooks/
│   └── AI_VIDEO_CREATOR.ipynb # Colab entry point
└── output/                   # generated scenes, images, audio, final MP4
```

## Configuration

All tunables live in `config.py`'s `Config` dataclass, and can be overridden
via environment variables (see `.env.example`) or by editing the file
directly:

- **Paths**: `output_dir` and subfolders are created automatically.
- **Device**: auto-detects CUDA vs CPU; `use_fp16` auto-disables on CPU.
- **Story model fallback chain**: `story_model_primary` →
  `story_model_fallbacks` (tried in order if a model fails to load, e.g. due
  to insufficient VRAM).
- **Quality presets**: `quality_preset = "draft" | "balanced" | "high"`
  trades off inference steps / resolution / fps for speed.
- **Video settings**: `scene_duration_seconds`, `transition_duration_seconds`,
  `fps`, `video_width/height`, `music_volume` (0.2-0.3 recommended).

## Memory management

Each pipeline stage (story, images, narration, music, video) loads only the
model(s) it needs and explicitly unloads + clears the CUDA cache before the
next stage starts (`utils.helpers.clear_gpu_memory`). On GPUs with less than
~12GB VRAM, the story model is loaded in 4-bit (via `bitsandbytes`) and the
image pipeline uses attention/VAE slicing automatically.

## Troubleshooting

- **Subtitles missing**: `TextClip` requires ImageMagick. The video assembler
  catches this and disables subtitles with a warning rather than failing the
  whole pipeline. Install ImageMagick (`apt-get install imagemagick`, and on
  some systems edit its policy.xml to allow text rendering) to enable them.
- **Story model OOM**: the pipeline automatically retries with each entry in
  `Config.story_model_fallbacks` (a Mistral base model, TinyLlama, then GPT-2)
  until one loads successfully.
- **Stable Audio Open unavailable**: requires `diffusers>=0.30`. If the
  pipeline can't load it, it logs a warning and falls back to a silent audio
  track so the rest of the pipeline still completes.
- **Gated models (401 errors)**: set `HF_TOKEN` in `.env` and accept the
  model license on its Hugging Face page first.
