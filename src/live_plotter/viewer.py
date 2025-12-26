from __future__ import annotations

import io
import queue
import threading
from typing import Iterable, Literal, Optional, Tuple

import numpy as np
from IPython.display import display
from PIL import Image
import ipywidgets as widgets

QueueItem = Tuple[str, object]
ImageFormat = Literal["jpeg", "png"]

_FORMAT_ALIASES = {
    "jpeg": "jpeg",
    "jpg": "jpeg",
    "png": "png",
}
_RESAMPLE = getattr(Image, "Resampling", Image)
_RESAMPLE_MAP = {
    "nearest": _RESAMPLE.NEAREST,
    "bilinear": _RESAMPLE.BILINEAR,
    "bicubic": _RESAMPLE.BICUBIC,
    "lanczos": _RESAMPLE.LANCZOS,
}

def _normalize_frame_for_video(frame: np.ndarray) -> np.ndarray:
    a = np.asarray(frame)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.ndim not in (2, 3):
        raise ValueError("Expected a 2D or 3D array for frame data.")
    if a.ndim == 3 and a.shape[-1] not in (3, 4):
        raise ValueError("Expected last dimension to be 3 (RGB) or 4 (RGBA).")
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    elif a.shape[-1] == 4:
        a = a[..., :3]
    return a


def _open_video_writer(record_path: str, *, fps: int, codec: Optional[str]):
    try:
        import imageio
    except ImportError as exc:
        raise ImportError(
            "Recording requires imageio. Install with: pip install imageio imageio-ffmpeg"
        ) from exc
    kwargs = {"fps": fps}
    if codec is not None:
        kwargs["codec"] = codec
    return imageio.get_writer(record_path, **kwargs)


def _normalize_format(image_format: str) -> ImageFormat:
    normalized = _FORMAT_ALIASES.get(image_format.lower())
    if normalized is None:
        allowed = ", ".join(sorted(_FORMAT_ALIASES))
        raise ValueError(f"Unsupported image_format '{image_format}'. Use one of: {allowed}.")
    return normalized  # type: ignore[return-value]


def _resolve_resample(resize_filter: str) -> int:
    key = resize_filter.lower()
    if key not in _RESAMPLE_MAP:
        allowed = ", ".join(sorted(_RESAMPLE_MAP))
        raise ValueError(f"Unsupported resize_filter '{resize_filter}'. Use one of: {allowed}.")
    return _RESAMPLE_MAP[key]


def encode_image(
    frame: np.ndarray,
    *,
    image_format: str = "jpeg",
    quality: int = 75,
    max_side: Optional[int] = None,
    keep_alpha: bool = True,
    png_compress_level: Optional[int] = None,
    png_optimize: bool = True,
    resize_filter: str = "bilinear",
) -> bytes:
    """Encode a frame as JPEG or PNG bytes.

    Args:
        frame: HxW or HxWxC array. Supports float in [0, 1] or uint8 in [0, 255].
        image_format: "jpeg" or "png" (case-insensitive).
        quality: JPEG quality setting (1-95 recommended).
        max_side: Optional maximum size for the longer edge.
        keep_alpha: Preserve alpha for PNG input when present.
        png_compress_level: PNG compression level (0-9). None keeps Pillow default.
        png_optimize: Enable PNG optimizer when True.
        resize_filter: Resize filter name (nearest, bilinear, bicubic, lanczos).

    Returns:
        Encoded image bytes.
    """
    normalized_format = _normalize_format(image_format)
    a = np.asarray(frame)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.ndim not in (2, 3):
        raise ValueError("Expected a 2D or 3D array for frame data.")
    if a.ndim == 3 and a.shape[-1] not in (3, 4):
        raise ValueError("Expected last dimension to be 3 (RGB) or 4 (RGBA).")
    if a.ndim == 2:
        if normalized_format == "jpeg":
            a = np.stack([a, a, a], axis=-1)
    elif a.shape[-1] == 4 and (normalized_format == "jpeg" or not keep_alpha):
        a = a[..., :3]

    img = Image.fromarray(a)
    if max_side is not None:
        w, h = img.size
        s = max(w, h)
        if s > max_side:
            resample = _resolve_resample(resize_filter)
            img = img.resize((int(w * max_side / s), int(h * max_side / s)), resample)

    buf = io.BytesIO()
    if normalized_format == "jpeg":
        img.save(buf, format="JPEG", quality=quality, optimize=True)
    else:
        save_kwargs = {"format": "PNG"}
        if png_compress_level is not None:
            save_kwargs["compress_level"] = png_compress_level
        if png_optimize:
            save_kwargs["optimize"] = True
        img.save(buf, **save_kwargs)
    return buf.getvalue()


def encode_jpeg(frame: np.ndarray, *, quality: int = 75, max_side: Optional[int] = None) -> bytes:
    """Encode a frame as JPEG bytes."""
    return encode_image(frame, image_format="jpeg", quality=quality, max_side=max_side, keep_alpha=False)


def _put_latest(q: "queue.Queue[QueueItem]", item: QueueItem) -> None:
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass


def loop(
    frame_gen: Iterable[np.ndarray],
    *,
    image_format: str = "jpeg",
    quality: int = 75,
    max_side: Optional[int] = 1024,
    keep_alpha: bool = True,
    png_compress_level: Optional[int] = None,
    png_optimize: bool = True,
    resize_filter: str = "bilinear",
    record_path: Optional[str] = None,
    record_fps: int = 30,
    record_codec: Optional[str] = None,
) -> None:
    """Display a live-updating generator of frames in a Jupyter cell."""
    q: "queue.Queue[QueueItem]" = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    def producer() -> None:
        writer = None
        try:
            if record_path is not None:
                writer = _open_video_writer(record_path, fps=record_fps, codec=record_codec)
            for f in frame_gen:
                if stop_event.is_set():
                    break
                if writer is not None:
                    writer.append_data(_normalize_frame_for_video(f))
                _put_latest(q, ("frame", f))
            _put_latest(q, ("done", None))
        except Exception as exc:  # pragma: no cover - re-raised in main thread
            _put_latest(q, ("error", exc))
        finally:
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    normalized_format = _normalize_format(image_format)
    w = widgets.Image(format=normalized_format)
    display(w)

    try:
        while True:
            kind, payload = q.get()
            if kind == "frame":
                with w.hold_sync():
                    w.value = encode_image(
                        payload,
                        image_format=normalized_format,
                        quality=quality,
                        max_side=max_side,
                        keep_alpha=keep_alpha,
                        png_compress_level=png_compress_level,
                        png_optimize=png_optimize,
                        resize_filter=resize_filter,
                    )
                continue
            if kind == "error":
                if isinstance(payload, BaseException):
                    raise payload
                raise RuntimeError("Live view failed with a non-exception error payload.")
            break
    finally:
        stop_event.set()
        thread.join(timeout=1.0)


def live_view_from_generator(
    frame_gen: Iterable[np.ndarray],
    *,
    image_format: str = "jpeg",
    quality: int = 75,
    max_side: Optional[int] = 1024,
    keep_alpha: bool = True,
    png_compress_level: Optional[int] = None,
    png_optimize: bool = True,
    resize_filter: str = "bilinear",
    record_path: Optional[str] = None,
    record_fps: int = 30,
    record_codec: Optional[str] = None,
) -> None:
    """Back-compat wrapper for older naming."""
    loop(
        frame_gen,
        image_format=image_format,
        quality=quality,
        max_side=max_side,
        keep_alpha=keep_alpha,
        png_compress_level=png_compress_level,
        png_optimize=png_optimize,
        resize_filter=resize_filter,
        record_path=record_path,
        record_fps=record_fps,
        record_codec=record_codec,
    )
