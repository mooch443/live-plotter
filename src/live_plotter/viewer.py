from __future__ import annotations

import io
import queue
import threading
from typing import Iterable, Optional, Tuple

import numpy as np
from IPython.display import display
from PIL import Image
import ipywidgets as widgets

QueueItem = Tuple[str, object]


def encode_jpeg(frame: np.ndarray, *, quality: int = 75, max_side: Optional[int] = None) -> bytes:
    """Encode a frame as JPEG bytes.

    Args:
        frame: HxW or HxWxC array. Supports float in [0, 1] or uint8 in [0, 255].
        quality: JPEG quality setting (1-95 recommended).
        max_side: Optional maximum size for the longer edge.

    Returns:
        JPEG-encoded bytes.
    """
    a = np.asarray(frame)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    if a.shape[-1] == 4:
        a = a[..., :3]

    img = Image.fromarray(a)
    if max_side is not None:
        w, h = img.size
        s = max(w, h)
        if s > max_side:
            img = img.resize((int(w * max_side / s), int(h * max_side / s)), Image.BILINEAR)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


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
    quality: int = 75,
    max_side: Optional[int] = 1024,
) -> None:
    """Display a live-updating generator of frames in a Jupyter cell."""
    q: "queue.Queue[QueueItem]" = queue.Queue(maxsize=1)

    def producer() -> None:
        try:
            for f in frame_gen:
                _put_latest(q, ("frame", f))
            _put_latest(q, ("done", None))
        except Exception as exc:  # pragma: no cover - re-raised in main thread
            _put_latest(q, ("error", exc))

    threading.Thread(target=producer, daemon=True).start()

    w = widgets.Image(format="jpeg")
    display(w)

    while True:
        kind, payload = q.get()
        if kind == "frame":
            with w.hold_sync():
                w.value = encode_jpeg(payload, quality=quality, max_side=max_side)
            continue
        if kind == "error":
            if isinstance(payload, BaseException):
                raise payload
            raise RuntimeError("Live view failed with a non-exception error payload.")
        break


def live_view_from_generator(
    frame_gen: Iterable[np.ndarray],
    *,
    quality: int = 75,
    max_side: Optional[int] = 1024,
) -> None:
    """Back-compat wrapper for older naming."""
    loop(frame_gen, quality=quality, max_side=max_side)
