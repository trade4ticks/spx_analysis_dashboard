"""A strategy's risk graph: an uploaded picture, checked and thumbnailed.

The page shows what the user uploaded; nothing here draws a payoff. A
calendar or a diagonal has no expiration curve worth computing, so the
picture is whatever the user's own platform drew.

prepare() is pure (bytes in, a dict out) and refuses anything that is not
a PNG, JPEG, WebP or GIF that Pillow can open -- the extension and the
browser's content type are not trusted, the decoded format is. The original
file is kept as uploaded; one THUMBNAIL is made here, once, sized for the
largest place the page shows a picture (a standalone card on a 2x screen),
so the calendar's mini cards never pull the full file.

Pillow is imported inside prepare(): matplotlib already depends on it, but a
host without it should still serve the page and every signal -- only the
upload is refused, and it says why.
"""
from __future__ import annotations

import io

MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 40_000_000           # a decompression bomb is refused, not decoded
THUMB_BOX = (480, 300)
FORMATS = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp", "GIF": "image/gif"}


def prepare(data: bytes) -> dict:
    if not data:
        raise ValueError("The file is empty.")
    if len(data) > MAX_BYTES:
        raise ValueError(f"The image is larger than {MAX_BYTES // (1024 * 1024)} MB.")
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError("Pillow is not installed on the server, so images cannot be uploaded.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            w, h = probe.size
            if w * h > MAX_PIXELS:
                raise ValueError(f"The image is {w} x {h} pixels -- too large to use.")
            probe.verify()
    except ValueError:
        raise
    except Exception:                                         # noqa: BLE001 -- anything Pillow cannot read
        raise ValueError("That file is not an image this page can read (PNG, JPEG, WebP or GIF).")
    if fmt not in FORMATS:
        raise ValueError(f"{fmt or 'That'} images are not accepted -- PNG, JPEG, WebP or GIF.")
    with Image.open(io.BytesIO(data)) as im:                  # verify() leaves the image unusable
        im.seek(0)
        im = im.convert("RGBA") if im.mode not in ("RGB", "RGBA") else im.copy()
    im.thumbnail(THUMB_BOX, Image.Resampling.LANCZOS)       # never enlarges; keeps the aspect ratio
    out = io.BytesIO()
    im.save(out, "WEBP", quality=88, method=4)
    return {"content_type": FORMATS[fmt], "data": bytes(data), "width": w, "height": h,
            "thumb": out.getvalue(), "thumb_type": "image/webp",
            "thumb_width": im.size[0], "thumb_height": im.size[1]}
