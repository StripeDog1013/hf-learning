"""
実行デバイス判定
"""

import torch


def get_device(cuda_id: int = 0) -> str:
    """
    利用可能なデバイスを返す

    Parameters
    ----------
    cuda_id : int
        使用するCUDAデバイス番号

    Returns
    -------
    str
        cuda:0 / cuda:1 / mps / cpu
    """

    if torch.cuda.is_available():

        gpu_count = torch.cuda.device_count()

        if cuda_id < 0 or cuda_id >= gpu_count:
            raise ValueError(
                f"Invalid cuda_id={cuda_id}. "
                f"Available range: 0 - {gpu_count - 1}"
            )

        return f"cuda:{cuda_id}"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def print_device_info(cuda_id: int = 0):

    device = get_device(cuda_id)

    print("=== Device Information ===")
    print(f"Selected Device: {device}")

    if device.startswith("cuda"):

        idx = int(device.split(":")[1])

        print(
            f"GPU Name: "
            f"{torch.cuda.get_device_name(idx)}"
        )

        props = torch.cuda.get_device_properties(idx)

        print(
            f"VRAM: "
            f"{props.total_memory / 1024**3:.2f} GB"
        )

    elif device == "mps":
        print("Apple MPS")

    else:
        print("CPU")
        
if __name__ == "__main__" :
    print_device_info()