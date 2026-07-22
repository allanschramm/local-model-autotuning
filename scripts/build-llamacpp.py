#!/usr/bin/env python3
"""build-llamacpp.py — Build llama.cpp binaries for CPU or CUDA/ROCm.

Automates configuring and building llama.cpp in build-cpu/ or build-cuda/.

Usage:
    python scripts/build-llamacpp.py --cpu
    python scripts/build-llamacpp.py --cuda
    python scripts/build-llamacpp.py --auto
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LLAMA_CPP_DIR = Path(os.environ.get("AUTORESEARCH_LLAMA_CPP_ROOT", REPO_ROOT / "llama.cpp"))


def is_gpu_available() -> bool:
    if platform.system() == "Darwin":
        return True
    smi = shutil.which("nvidia-smi") or shutil.which("rocm-smi")
    if not smi:
        return False
    try:
        flag = "-L" if "nvidia" in smi.lower() else "-i"
        res = subprocess.run([smi, flag], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return res.returncode == 0
    except Exception:
        return False


def build_llamacpp(target: str, clean: bool = False) -> int:
    if not LLAMA_CPP_DIR.exists():
        print(f"ERROR: llama.cpp directory not found at {LLAMA_CPP_DIR}", file=sys.stderr)
        print("Please clone llama.cpp first: git clone https://github.com/ggerganov/llama.cpp.git", file=sys.stderr)
        return 1

    build_dir_name = "build-cuda" if target == "cuda" else "build-cpu"
    build_dir = LLAMA_CPP_DIR / build_dir_name

    if clean and build_dir.exists():
        print(f"Cleaning build directory: {build_dir}")
        shutil.rmtree(build_dir, ignore_errors=True)

    cmake_args = [
        "cmake",
        "-S", str(LLAMA_CPP_DIR),
        "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_BUILD_SERVER=ON",
    ]

    if shutil.which("ninja"):
        cmake_args.extend(["-G", "Ninja"])

    if target == "cuda":
        cmake_args.append("-DGGML_CUDA=ON")
    else:
        cmake_args.extend(["-DGGML_CUDA=OFF", "-DGGML_NATIVE=ON"])

    print(f"Configuring llama.cpp ({target.upper()} build in {build_dir_name})...")
    res = subprocess.run(cmake_args)
    if res.returncode != 0:
        print(f"ERROR: CMake configure failed with code {res.returncode}", file=sys.stderr)
        return res.returncode

    jobs = str(os.cpu_count() or 4)
    build_args = ["cmake", "--build", str(build_dir), "--config", "Release", "-j", jobs]
    print(f"Building llama.cpp ({target.upper()})...")
    res = subprocess.run(build_args)
    if res.returncode == 0:
        print(f"SUCCESS: llama.cpp ({target.upper()}) build completed successfully!")
    return res.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cpu", action="store_true", help="build CPU-only binary target (default)")
    group.add_argument("--cuda", action="store_true", help="build CUDA GPU binary target")
    group.add_argument("--auto", action="store_true", help="auto-detect hardware and build appropriate target")
    parser.add_argument("--clean", action="store_true", help="clean build directory before building")

    args = parser.parse_args()

    if args.cuda:
        target = "cuda"
    elif args.auto:
        if platform.system() == "Darwin":
            target = "cpu"
            print("Auto-detected macOS target (CPU / Native Metal)")
        elif is_gpu_available():
            target = "cuda"
            print("Auto-detected GPU target: CUDA")
        else:
            target = "cpu"
            print("Auto-detected CPU target")
    else:
        target = "cpu"

    return build_llamacpp(target, clean=args.clean)


if __name__ == "__main__":
    sys.exit(main())
