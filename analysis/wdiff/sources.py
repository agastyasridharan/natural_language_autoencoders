"""Tensor sources: stream individual tensors from HF repos or local dirs.

Two implementations of the same interface:

- RemoteSafetensorsRepo — parses safetensors headers via HTTP range requests
  and fetches each tensor's exact byte range. Nothing is written to disk.
  (Required here: the analysis host has ~5 GB free disk, far less than any
  checkpoint.)
- LocalSafetensorsRepo — safetensors.safe_open over a local directory.

Both build a name -> (file, dtype, shape, offsets) index from
model.safetensors.index.json when present (single-file and extra sidecar
.safetensors files such as the AR's value_head.safetensors are handled too),
and return tensors upcast to float32 on CPU.
"""
from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import torch

_TORCH_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}

INDEX_NAME = "model.safetensors.index.json"


@dataclass(frozen=True)
class TensorMeta:
    name: str
    file: str            # shard filename (repo-relative)
    dtype: str           # safetensors dtype string, e.g. "BF16"
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]  # relative to shard data section
    data_start: int      # absolute byte offset of the shard's data section

    @property
    def nbytes(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]


def _parse_header(blob: bytes) -> tuple[dict, int]:
    """Parse a safetensors header. `blob` must start at byte 0 of the file
    and contain at least the full JSON header. Returns (header, data_start)."""
    (n,) = struct.unpack("<Q", blob[:8])
    header = json.loads(blob[8 : 8 + n])
    return header, 8 + n


def _decode(buf: bytearray, dtype: str, shape: tuple[int, ...]) -> torch.Tensor:
    t = torch.frombuffer(memoryview(buf), dtype=_TORCH_DTYPES[dtype])
    t = t.reshape(shape)
    if t.dtype in (torch.float16, torch.bfloat16, torch.float64):
        t = t.to(torch.float32)
    return t


class _BaseRepo:
    """Shared index logic. Subclasses provide _read_header(file) and
    _fetch(meta)."""

    def __init__(self) -> None:
        self._meta: dict[str, TensorMeta] = {}
        self.bytes_fetched = 0

    # -- index construction ------------------------------------------------
    def _build_index(self, st_files: list[str]) -> None:
        # Shard headers are authoritative (model.safetensors.index.json is
        # advisory and misses extras like value_head.safetensors). Headers are
        # small (tens of KB); read them all up front so we can order the run
        # by (shard, offset) and report exact planned bytes.
        for f in sorted(set(st_files)):
            header, data_start = self._read_header(f)
            for name, ent in header.items():
                if name == "__metadata__":
                    continue
                self._meta[name] = TensorMeta(
                    name=name,
                    file=f,
                    dtype=ent["dtype"],
                    shape=tuple(ent["shape"]),
                    data_offsets=tuple(ent["data_offsets"]),
                    data_start=data_start,
                )

    # -- public interface ----------------------------------------------------
    def names(self) -> list[str]:
        return list(self._meta)

    def meta(self, name: str) -> TensorMeta:
        return self._meta[name]

    def tensor(self, name: str) -> torch.Tensor:
        m = self._meta[name]
        buf = self._fetch(m)
        if len(buf) != m.nbytes:
            raise IOError(f"{self.label}:{name}: got {len(buf)} bytes, want {m.nbytes}")
        self.bytes_fetched += m.nbytes
        return _decode(buf, m.dtype, m.shape)


class RemoteSafetensorsRepo(_BaseRepo):
    def __init__(self, repo_id: str, token: str | None = None, revision: str = "main"):
        super().__init__()
        import httpx  # repo-standard HTTP client

        self.repo_id = repo_id
        self.label = repo_id
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._client = httpx.Client(
            headers=headers,
            timeout=httpx.Timeout(30.0, read=300.0),
            follow_redirects=True,
        )
        # Pin to a commit sha so a mid-run repo update can't skew the diff.
        r = self._get(f"https://huggingface.co/api/models/{repo_id}/revision/{revision}")
        info = r.json()
        self.sha = info["sha"]
        files = [s["rfilename"] for s in info["siblings"]]
        st_files = [f for f in files if f.endswith(".safetensors") and "/" not in f]
        if not st_files:
            raise FileNotFoundError(f"{repo_id}: no .safetensors files")
        self._build_index(st_files)

    def _url(self, fname: str) -> str:
        return f"https://huggingface.co/{self.repo_id}/resolve/{self.sha}/{fname}"

    def _get(self, url: str, expect_partial: bool = False, **kw):
        last = None
        for attempt in range(4):
            try:
                r = self._client.get(url, **kw)
                if expect_partial and r.status_code == 200:
                    # a 200 on a ranged request means the server ignored Range
                    # and sent the whole (multi-GB) file — fail loudly
                    raise IOError(f"server ignored Range header for {url}")
                if r.status_code in (200, 206):
                    return r
                if r.status_code in (401, 403, 404):
                    raise PermissionError(f"HTTP {r.status_code} for {url}")
                last = IOError(f"HTTP {r.status_code} for {url}")
            except PermissionError:
                raise
            except Exception as e:  # noqa: BLE001 - retry network errors
                last = e
            time.sleep(2.0 ** attempt)
        raise last

    def _read_header(self, fname: str) -> tuple[dict, int]:
        r = self._get(self._url(fname), expect_partial=True,
                      headers={"Range": "bytes=0-7"})
        (n,) = struct.unpack("<Q", r.content)
        r = self._get(self._url(fname), expect_partial=True,
                      headers={"Range": f"bytes=8-{8 + n - 1}"})
        return json.loads(r.content), 8 + n

    def _fetch(self, m: TensorMeta) -> bytearray:
        a = m.data_start + m.data_offsets[0]
        b = m.data_start + m.data_offsets[1] - 1
        buf = bytearray(m.nbytes)
        view = memoryview(buf)
        pos = 0
        last = None
        for attempt in range(4):
            try:
                pos = 0
                with self._client.stream(
                    "GET", self._url(m.file), headers={"Range": f"bytes={a}-{b}"}
                ) as resp:
                    if resp.status_code != 206:
                        raise IOError(f"HTTP {resp.status_code} (need 206) fetching {m.name}")
                    for chunk in resp.iter_bytes(8 * 1024 * 1024):
                        view[pos : pos + len(chunk)] = chunk
                        pos += len(chunk)
                if pos == m.nbytes:
                    return buf
                last = IOError(f"{m.name}: short read {pos}/{m.nbytes}")
            except Exception as e:  # noqa: BLE001
                last = e
            time.sleep(2.0 ** attempt)
        raise last


class LocalSafetensorsRepo(_BaseRepo):
    def __init__(self, root: str | Path):
        super().__init__()
        self.root = Path(root)
        self.label = str(root)
        self.sha = "local"
        st_files = sorted(p.name for p in self.root.glob("*.safetensors"))
        if not st_files:
            raise FileNotFoundError(f"{root}: no .safetensors files")
        self._build_index(st_files)

    def _read_header(self, fname: str) -> tuple[dict, int]:
        with open(self.root / fname, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            return json.loads(f.read(n)), 8 + n

    def _fetch(self, m: TensorMeta) -> bytearray:
        raise NotImplementedError("local repo reads via safe_open in tensor()")

    def tensor(self, name: str) -> torch.Tensor:
        from safetensors import safe_open

        m = self._meta[name]
        with safe_open(str(self.root / m.file), framework="pt", device="cpu") as f:
            t = f.get_tensor(name)
        self.bytes_fetched += m.nbytes
        # clone when already fp32 so the runner's in-place ops never touch
        # safetensors-owned (possibly read-only) memory
        t = t.to(torch.float32) if t.dtype != torch.float32 else t.clone()
        return t


def open_repo(spec: str, token: str | None = None, revision: str = "main") -> _BaseRepo:
    """`spec` is a local directory or a HF repo id."""
    if Path(spec).expanduser().is_dir():
        return LocalSafetensorsRepo(Path(spec).expanduser())
    return RemoteSafetensorsRepo(spec, token=token, revision=revision)
