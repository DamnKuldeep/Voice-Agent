"""Environment probe: GPU, driver, free VRAM and library versions. Exits 1 if torch cannot use CUDA."""

from __future__ import annotations

import importlib.metadata as md
import subprocess
import sys

LIBRARIES = [
    "torch", "torchaudio", "transformers", "huggingface_hub", "kokoro", "misaki",
    "silero-vad", "onnxruntime", "fastrtc", "gradio", "numpy", "soundfile",
]


def smi(*query: str) -> str:
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(query)}", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    )
    return out.stdout.strip()


def version(name: str) -> str:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "not installed"


def main() -> int:
    print("python", sys.version.split()[0])
    print("nvidia-smi:", smi("name", "driver_version", "memory.total", "memory.used", "memory.free"))
    for lib in LIBRARIES:
        print(f"{lib:16s} {version(lib)}")
    import torch

    print("torch cuda build:", torch.version.cuda, "| available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("FAIL: torch cannot see the GPU")
        return 1
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    print(f"device: {props.name}, capability {props.major}.{props.minor}, sm count {props.multi_processor_count}")
    print(f"VRAM total {total / 2**20:.0f} MiB, free {free / 2**20:.0f} MiB (before any model is loaded)")
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    print("fp16 matmul ok:", bool((x @ x).isfinite().all()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
