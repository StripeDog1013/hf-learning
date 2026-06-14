import torch


def get_device(cuda_id: int = 0) -> str:

    if torch.cuda.is_available():

        gpu_count = torch.cuda.device_count()

        if cuda_id < 0 or cuda_id >= gpu_count:
            raise ValueError(
                f"Invalid cuda_id={cuda_id}"
            )

        return f"cuda:{cuda_id}"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def get_torch_dtype(device: str):
    if device.startswith("cuda"):
        return torch.float16

    if device == "mps":
        return torch.float16

    return torch.float32


def print_device_info(cuda_id: int = 0) -> None:
    device = get_device(cuda_id)

    print("\n=== Device Information ===")
    print(f"Selected Device: {device}")

    if device.startswith("cuda"):
        idx = int(device.split(":")[1])

        print(f"GPU Name : {torch.cuda.get_device_name(idx)}")

        props = torch.cuda.get_device_properties(idx)

        print(
            f"VRAM     : "
            f"{props.total_memory / 1024**3:.2f} GB"
        )

    elif device == "mps":
        print("Apple MPS")

    else:
        print("CPU")