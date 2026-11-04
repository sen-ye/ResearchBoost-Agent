"""SHA-256 of tensors off the critical path: the same digests as hashlib.sha256 over each tensor's contiguous bytes.

The judge's bitwise records hash every recorded tensor (parameters, gradients, optimizer state; tens of GB per rank
on the larger tasks). Done one tensor after the other, each hash waits for a synchronous device-to-host copy and then
for one CPU core. ShaPipeline overlaps the three: a CUDA tensor is copied into pinned host memory asynchronously on
the current stream (later work on that stream, including the reuse of the tensor's memory after the caller drops
it, is ordered after the copy), and the digest is computed in a thread pool once the copy's event has completed
(hashlib releases the GIL on large buffers, so the threads hash in parallel). A CPU tensor is cloned first, so that
the caller may change it right away. At most `inflight_bytes` of copies are held; submit() waits for the oldest
digests beyond that. Digests are those of the plain formulation, `hashlib.sha256(memoryview(t.contiguous()
.reshape(-1).view(torch.uint8).numpy())).hexdigest()`, so frozen records stay valid.

Task benches copy numerics/ as is (goal/templates/FORMULATOR_BRIEF.md), this file included.
"""
import collections
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os

import torch

INFLIGHT_BYTES = 2 << 30
MAX_WORKERS = 8


def default_workers():
    """This process's CPU share (the bench pins every rank to its share), at most MAX_WORKERS threads."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    return max(1, min(MAX_WORKERS, cpus))


def _flat(t):
    return t.detach().contiguous().reshape(-1)


def sha_host(host):
    """The digest of a flat host tensor's bytes."""
    if not host.numel():
        return hashlib.sha256(b'').hexdigest()
    return hashlib.sha256(memoryview(host.view(torch.uint8).numpy())).hexdigest()


def tensor_sha(t):
    """Synchronous digest of one tensor (the reference formulation)."""
    return sha_host(_flat(t).cpu())


def _wait_and_hash(host, event):
    if event is not None:
        event.synchronize()
    return sha_host(host)


class ShaPipeline:
    """submit(tensor) -> Future of its SHA-256 hex digest; close() (or the with block) waits for all of them."""

    def __init__(self, workers=None, inflight_bytes=INFLIGHT_BYTES):
        self.pool = ThreadPoolExecutor(workers or default_workers(), thread_name_prefix='sha')
        self.inflight_bytes = inflight_bytes
        self.pending = collections.deque()
        self.held = 0

    def _drain(self, need):
        while self.pending and self.held + need > self.inflight_bytes:
            future, nbytes = self.pending.popleft()
            future.result()
            self.held -= nbytes

    def submit(self, t):
        flat = _flat(t)
        nbytes = flat.numel() * flat.element_size()
        self._drain(nbytes)
        if flat.is_cuda:
            host = torch.empty(flat.shape, dtype=flat.dtype, pin_memory=True)
            host.copy_(flat, non_blocking=True)
            event = torch.cuda.Event()
            event.record()
        else:
            host, event = flat.clone(), None
        future = self.pool.submit(_wait_and_hash, host, event)
        self.pending.append((future, nbytes))
        self.held += nbytes
        return future

    def close(self):
        self.pool.shutdown(wait=True)
        self.pending.clear()
        self.held = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def tensor_shas(tensors, **kwargs):
    """Digests of an iterable of tensors, in order, through one pipeline."""
    with ShaPipeline(**kwargs) as pipe:
        futures = [pipe.submit(t) for t in tensors]
        return [f.result() for f in futures]
