import json
import subprocess
import sys

import torch


def run_nvidia_smi():
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,cuda_version", "--format=csv,noheader"], text=True)
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        return {"ok": True, "gpus": lines}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def main():
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda_compiled": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
    }

    if torch.cuda.is_available():
        info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    info["nvidia_smi"] = run_nvidia_smi()

    mismatch = False
    if info["cuda_available"] and info["torch_cuda_compiled"] is None:
        mismatch = True
    info["potential_cuda_mismatch"] = mismatch

    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
