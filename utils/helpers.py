"""Small, dependency-light helpers shared by every pipeline module."""

import gc
import json
import re
import time
import functools
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, TypeVar

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger("ai_video_creator.helpers")

T = TypeVar("T")


def clear_gpu_memory() -> None:
    """Release cached CUDA memory and run the Python garbage collector.

    Call this between pipeline stages (story -> image -> audio -> video) so
    that each stage gets the maximum VRAM available, since Colab GPUs are
    frequently memory constrained (T4 = 15GB).
    """
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


@contextmanager
def timer(label: str, log: logging.Logger = None):
    """Context manager that logs how long a block of code took to run."""
    log = log or logger
    start = time.time()
    log.info("-> starting: %s", label)
    try:
        yield
    finally:
        elapsed = time.time() - start
        log.info("<- finished: %s (%.1fs)", label, elapsed)


def retry(max_attempts: int = 3, delay_seconds: float = 2.0, backoff: float = 2.0):
    """Decorator that retries a function on exception, with exponential backoff.

    Used around model downloads/loads and generation calls, which can fail
    transiently in Colab due to flaky network access or OOM spikes that
    resolve after a GPU cache clear.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            attempt = 0
            current_delay = delay_seconds
            last_exc = None
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - intentional broad catch for retry
                    last_exc = exc
                    attempt += 1
                    logger.warning(
                        "%s failed (attempt %d/%d): %s",
                        func.__name__,
                        attempt,
                        max_attempts,
                        exc,
                    )
                    if attempt < max_attempts:
                        clear_gpu_memory()
                        time.sleep(current_delay)
                        current_delay *= backoff
            raise last_exc

        return wrapper

    return decorator


def save_json(data: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(text: str, max_length: int = 60) -> str:
    """Turn arbitrary theme text into a filesystem-safe slug."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_length] or "untitled"
