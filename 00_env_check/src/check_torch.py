import torch

print("=== PyTorch Check ===")
print(f"PyTorch version : {torch.__version__}")

print("\n--- CUDA ---")
print(f"CUDA available : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version   : {torch.version.cuda}")
    print(f"GPU count      : {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        print(f"[{i}] {torch.cuda.get_device_name(i)}")

        props = torch.cuda.get_device_properties(i)
        total_vram_gb = props.total_memory / 1024**3
        print(f"    VRAM: {total_vram_gb:.2f} GB")

print("\n--- MPS ---")
print(f"MPS available  : {torch.backends.mps.is_available()}")
print(f"MPS built      : {torch.backends.mps.is_built()}")

print("\n--- Tensor Test ---")
x = torch.randn(3, 3)
print(x)
print("CPU tensor OK")

if torch.cuda.is_available():
    x_cuda = x.to("cuda")
    print("CUDA tensor OK")
    print(x_cuda)

if torch.backends.mps.is_available():
    x_mps = x.to("mps")
    print("MPS tensor OK")
    print(x_mps)