"""GPU residency check and memory readings shared by the adapters, benchmarks and the UI."""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Iterable
from typing import Self

import torch


def apply_cpu_threads(count: int) -> None:
    """Limit torch's CPU pool; the default of all cores oversubscribes next to the decode and audio threads."""
    torch.set_num_threads(count)


class ResidencyError(RuntimeError):
    """A model parameter or buffer sits somewhere other than its configured device."""


def assert_resident(name: str, modules: Iterable[torch.nn.Module], device: str) -> int:
    """Fail loudly if any parameter or buffer is off ``device``. Returns the tensor count checked."""
    want = torch.device(device)
    count = 0
    for module in modules:
        tensors = [*module.named_parameters(), *module.named_buffers()]
        for tensor_name, tensor in tensors:
            count += 1
            if tensor.device.type != want.type or tensor.device.index not in (None, want.index):
                raise ResidencyError(f"{name}: {tensor_name} is on {tensor.device}, expected {want}")
    if count == 0:
        raise ResidencyError(f"{name}: no tensors found to check")
    return count


def process_memory_mib(device: str = "cuda:0") -> dict[str, float]:
    """This process's allocator view: currently allocated, peak allocated, peak reserved (MiB)."""
    mib = 2**20
    return {
        "allocated": torch.cuda.memory_allocated(device) / mib,
        "peak_allocated": torch.cuda.max_memory_allocated(device) / mib,
        "peak_reserved": torch.cuda.max_memory_reserved(device) / mib,
    }


class ClockSampler:
    """Samples SM clock, power, temperature and throttle reasons once a second while a benchmark runs."""

    FIELDS = "utilization.gpu,clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active"

    def __init__(self, index: int = 0) -> None:
        self._index = index
        self._rows: list[list[float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="gpu-clocks")

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.is_set():
            out = subprocess.run(
                ["nvidia-smi", f"--id={self._index}", f"--query-gpu={self.FIELDS}",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=False,
            ).stdout.strip()
            try:
                util, clock, power, temp, reasons = [x.strip() for x in out.split(",")]
                self._rows.append([float(util), float(clock), float(power), float(temp), int(reasons, 16)])
            except ValueError:
                pass
            self._stop.wait(0.5)

    def summary(self) -> str:
        busy = [r for r in self._rows if r[0] >= 30] or self._rows
        if not busy:
            return "no clock samples"
        clocks = sorted(r[1] for r in busy)
        reasons = sorted({hex(int(r[4])) for r in busy})
        return (f"SM clock while busy (n={len(busy)}): median {clocks[len(clocks) // 2]:.0f} MHz, "
                f"min {clocks[0]:.0f}; max temp {max(r[3] for r in busy):.0f} C; "
                f"max power {max(r[2] for r in busy):.0f} W; throttle reasons seen {reasons}")


def device_memory_mib(index: int = 0) -> dict[str, float]:
    """Whole-GPU used/total in MiB from nvidia-smi: includes the CUDA context and other programs."""
    out = subprocess.run(
        ["nvidia-smi", f"--id={index}", "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.split(",")
    return {"used": float(out[0]), "total": float(out[1])}
