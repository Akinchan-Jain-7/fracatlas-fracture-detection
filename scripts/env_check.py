"""Print the runtime environment: Python, torch, CUDA/GPU, platform, data path.

Paste this output whenever reporting a run. Most "it works on my machine"
bugs are version or device differences visible here.
"""

from __future__ import annotations

import os
import platform
import sys

from fracdet.config import load_config
from fracdet.paths import find_dataset_root, is_kaggle, resolve_data_dir


def main() -> None:
    print(f"python      : {sys.version.split()[0]} ({platform.platform()})")
    print(f"cpu cores   : {os.cpu_count()}")
    print(f"on kaggle   : {is_kaggle()}")
    try:
        import torch

        print(f"torch       : {torch.__version__} (cuda build: {torch.version.cuda})")
        print(f"cuda avail  : {torch.cuda.is_available()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  gpu[{i}]    : {props.name}, {props.total_memory / 1024**3:.1f} GiB")
    except ImportError:
        print("torch       : NOT INSTALLED")

    cfg = load_config("configs/base.yaml")
    data_dir = resolve_data_dir(cfg)
    print(f"data dir    : {data_dir}")
    print(f"dataset root: {find_dataset_root(data_dir)}")


if __name__ == "__main__":
    main()
