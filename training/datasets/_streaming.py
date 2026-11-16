"""Shared plumbing for the streaming food datasets.

All three datasets work the same way: the (small) label metadata is fetched
once and cached on disk, the images are fetched over HTTP while iterating and
are never stored unless ``image_cache_dir`` is set.

Every sample is a dict with at least::

    image       PIL.Image (RGB), or whatever ``transform`` returns
    dataset     dataset name
    dish_id     split unit: all photos of one dish share it
    image_id    unique per photo
    mass_g, kcal, fat_g, carb_g, protein_g   floats (nan if unknown)
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
import os
import random
import struct
import threading
import zipfile
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterator

import requests
from PIL import Image, ImageOps
from requests.adapters import HTTPAdapter
from torch.utils.data import IterableDataset, get_worker_info
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

CACHE_DIR = Path(os.environ.get("FOOD_DATA_CACHE", Path.home() / ".cache" / "ondevice_food"))
TARGET_KEYS = ("mass_g", "kcal", "fat_g", "carb_g", "protein_g")

_local = threading.local()
_url_lock = threading.Lock()


# --------------------------------------------------------------------- HTTP --

def session() -> requests.Session:
    """One retrying session per thread and process (safe across DataLoader workers)."""
    if getattr(_local, "pid", None) != os.getpid():
        s = requests.Session()
        retry = Retry(total=4, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=("GET",))
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.headers["User-Agent"] = "food-dataset-streamer"
        _local.s, _local.pid = s, os.getpid()
    return _local.s


def fetch(url: str, headers: dict | None = None, timeout: float = 60) -> bytes:
    for attempt in range(3):
        try:
            r = session().get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.content
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError):
            if attempt == 2:  # transfers of large files break off now and then
                raise
    raise RuntimeError("unreachable")


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def cached(path: Path, produce: Callable[[], bytes]) -> bytes:
    """Return the file's content, producing and storing it on first use."""
    if path.exists():
        return path.read_bytes()
    data = produce()
    _write_atomic(path, data)
    return data


def hash_split(key: str, fractions: tuple[float, ...], seed: int = 0) -> int:
    """Deterministically assign ``key`` (a dish id) to one of ``len(fractions)`` parts."""
    h = int.from_bytes(hashlib.sha1(f"{seed}:{key}".encode()).digest()[:8], "big") / 2**64
    acc = 0.0
    for i, f in enumerate(fractions):
        acc += f
        if h < acc:
            return i
    return len(fractions) - 1


def to_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


# --------------------------------------------------------------- remote zip --

class _RangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file, one range request per read."""

    def __init__(self, read_range: Callable[[int, int], bytes], size: int):
        self._read_range, self._size, self._pos = read_range, size, 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self._pos

    def seek(self, offset, whence=0):
        self._pos = offset if whence == 0 else (self._pos if whence == 1 else self._size) + offset
        return self._pos

    def readinto(self, b):
        n = min(len(b), self._size - self._pos)
        if n <= 0:
            return 0
        data = self._read_range(self._pos, self._pos + n)
        b[:len(data)] = data
        self._pos += len(data)
        return len(data)


class RemoteZip:
    """Random access to members of a zip archive on an HTTP server (range requests).

    ``resolve`` returns a (possibly short-lived, signed) URL of the archive; it
    is called again when the URL stops working.
    """

    def __init__(self, resolve: Callable[[], str]):
        self._resolve = resolve
        self._url: str | None = None
        self.size = self._probe_size()

    def _probe_size(self) -> int:
        r = self._get({"Range": "bytes=0-0"})
        return int(r.headers["Content-Range"].rsplit("/", 1)[1])

    def _get(self, headers: dict) -> requests.Response:
        for attempt in range(2):
            with _url_lock:
                if self._url is None:
                    self._url = self._resolve()
                url = self._url
            r = session().get(url, headers=headers, timeout=60)
            if r.status_code in (400, 401, 403) and attempt == 0:  # signed URL expired
                with _url_lock:
                    if self._url == url:
                        self._url = None
                continue
            r.raise_for_status()
            return r
        raise RuntimeError("unreachable")

    def read_range(self, start: int, end: int) -> bytes:
        """Bytes ``[start, end)`` of the archive."""
        return self._get({"Range": f"bytes={start}-{end - 1}"}).content

    def index(self) -> dict[str, list[int]]:
        """``{member name: [header offset, compressed size, method]}`` (3 requests)."""
        with zipfile.ZipFile(io.BufferedReader(_RangeFile(self.read_range, self.size), 1 << 20)) as z:
            return {i.filename: [i.header_offset, i.compress_size, i.compress_type]
                    for i in z.infolist() if not i.is_dir()}

    def read(self, name: str, entry: list[int]) -> bytes:
        """Content of one member, normally with a single range request."""
        offset, csize, method = entry
        guess = 30 + len(name.encode()) + 128  # local header + name + room for the extra field
        buf = self.read_range(offset, min(self.size, offset + guess + csize))
        if buf[:4] != b"PK\x03\x04":
            raise OSError(f"bad zip local header for {name}")
        name_len, extra_len = struct.unpack_from("<HH", buf, 26)
        start = 30 + name_len + extra_len
        if len(buf) < start + csize:
            buf += self.read_range(offset + len(buf), offset + start + csize)
        data = buf[start:start + csize]
        if method == zipfile.ZIP_STORED:
            return data
        if method == zipfile.ZIP_DEFLATED:
            return zlib.decompressobj(-15).decompress(data)
        raise OSError(f"unsupported zip compression {method} for {name}")


# ------------------------------------------------------------- base dataset --

class StreamingFoodDataset(IterableDataset):
    """Iterable dataset over ``self.records``; images are fetched while iterating.

    Args:
        transform: applied to the RGB PIL image.
        shuffle: shuffle the full record list each epoch (call ``set_epoch``).
        seed: seed for shuffling.
        prefetch: download threads per (worker) process; samples keep their order.
        image_cache_dir: if set, downloaded image bytes are kept there and reused.
        draft_size: hint for the JPEG decoder to decode at reduced resolution
            (shorter side stays >= draft_size). Makes 12 MP phone photos ~4x faster.
        skip_errors: skip (and log) samples whose image cannot be loaded.
        shuffle_buffer: with ``shuffle``, additionally mix the sample stream through
            a buffer of this size. Only needed when one record yields several
            samples (video frames), which would otherwise arrive back to back.
    """

    name = ""

    def __init__(self, transform: Callable | None = None, shuffle: bool = False, seed: int = 0,
                 prefetch: int = 8, image_cache_dir: str | os.PathLike | None = None,
                 draft_size: int | None = None, skip_errors: bool = True,
                 shuffle_buffer: int = 0):
        self.transform = transform
        self.shuffle = shuffle
        self.seed = seed
        self.prefetch = prefetch
        self.image_cache_dir = Path(image_cache_dir) if image_cache_dir else None
        self.draft_size = draft_size
        self.skip_errors = skip_errors
        self.shuffle_buffer = shuffle_buffer
        self.epoch = 0
        self.records: list[dict] = []

    # -- to implement ---------------------------------------------------------

    def _load(self, rec: dict) -> dict | list[dict]:
        """Fetch the image(s) of ``rec`` and return the finished sample (or several)."""
        raise NotImplementedError

    # -- helpers for subclasses -----------------------------------------------

    def _bytes(self, key: str, produce: Callable[[], bytes]) -> bytes:
        if self.image_cache_dir is None:
            return produce()
        return cached(self.image_cache_dir / self.name / key, produce)

    def _image(self, key: str, produce: Callable[[], bytes]) -> Image.Image:
        img = Image.open(io.BytesIO(self._bytes(key, produce)))
        if self.draft_size:
            img.draft("RGB", (self.draft_size, self.draft_size))
        return ImageOps.exif_transpose(img).convert("RGB")  # phone photos carry their rotation in EXIF

    def _sample(self, rec: dict, image: Image.Image, **extra) -> dict:
        sample = {k: v for k, v in rec.items() if not k.startswith("_")}
        sample["image"] = self.transform(image) if self.transform else image
        sample["dataset"] = self.name
        sample.update(extra)
        return sample

    # -- iteration ------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def _safe_load(self, rec: dict) -> list[dict]:
        try:
            out = self._load(rec)
            return out if isinstance(out, list) else [out]
        except Exception as e:  # noqa: BLE001 - dead links / broken files must not kill training
            if not self.skip_errors:
                raise
            log.warning("%s: skipping %s (%s: %s)", self.name, rec.get("image_id"), type(e).__name__, e)
            return []

    def __iter__(self) -> Iterator[dict]:
        n = self.shuffle_buffer if self.shuffle else 0
        if n <= 1:
            yield from self._stream()
            return
        info = get_worker_info()
        rng = random.Random(f"{self.seed}-{self.epoch}-{info.id if info else 0}-buffer")
        buffer: list[dict] = []
        for sample in self._stream():
            if len(buffer) < n:
                buffer.append(sample)
            else:
                i = rng.randrange(n)
                buffer[i], sample = sample, buffer[i]
                yield sample
        rng.shuffle(buffer)
        yield from buffer

    def _stream(self) -> Iterator[dict]:
        records = self.records
        if self.shuffle:
            records = list(records)
            random.Random(f"{self.seed}-{self.epoch}").shuffle(records)
        info = get_worker_info()
        if info is not None:
            records = records[info.id::info.num_workers]

        if self.prefetch <= 1:
            for rec in records:
                yield from self._safe_load(rec)
            return

        pool = ThreadPoolExecutor(self.prefetch, thread_name_prefix=f"{self.name}-fetch")
        pending: deque = deque()
        try:
            for rec in records:
                pending.append(pool.submit(self._safe_load, rec))
                if len(pending) >= 2 * self.prefetch:
                    yield from pending.popleft().result()
            while pending:
                yield from pending.popleft().result()
        finally:
            pool.shutdown(wait=False, cancel_futures=True)


def collate_samples(batch: list[dict]) -> dict:
    """Collate for ``DataLoader``: stacks tensor images, builds a ``targets`` tensor
    ``[B, 5]`` in the order of ``TARGET_KEYS`` and keeps everything else as lists."""
    import torch

    out: dict[str, Any] = {k: [s.get(k) for s in batch] for k in batch[0]}
    if all(isinstance(x, torch.Tensor) for x in out["image"]):
        out["image"] = torch.stack(out["image"])
    out["targets"] = torch.tensor([[s[k] for k in TARGET_KEYS] for s in batch], dtype=torch.float32)
    return out


class Interleave(IterableDataset):
    """Mix several streaming datasets into one stream, each drawn in proportion to its size."""

    def __init__(self, datasets: list[StreamingFoodDataset], seed: int = 0):
        self.datasets, self.seed, self.epoch = list(datasets), seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        for d in self.datasets:
            d.set_epoch(epoch)

    def __len__(self) -> int:
        return sum(len(d) for d in self.datasets)

    def __iter__(self) -> Iterator[dict]:
        info = get_worker_info()
        rng = random.Random(f"{self.seed}-{self.epoch}-{info.id if info else 0}-mix")
        streams = [(iter(d), len(d)) for d in self.datasets if len(d)]
        while streams:
            i = rng.choices(range(len(streams)), weights=[w for _, w in streams])[0]
            try:
                yield next(streams[i][0])
            except StopIteration:
                del streams[i]
