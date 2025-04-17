import torch

def check_gpus():
    if not torch.cuda.is_available():
        print("CUDA is not available. No GPU detected.")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"Total GPUs detected: {num_gpus}")

    for i in range(num_gpus):
        try:
            device = torch.device(f"cuda:{i}")
            torch.cuda.set_device(device)
            _ = torch.tensor([1.0], device=device)  # 在 GPU 上创建一个张量
            
            name = torch.cuda.get_device_name(i)
            capability = torch.cuda.get_device_capability(i)
            memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # 转换为 GB
            
            print(f"GPU {i}: {name}")
            print(f"  Compute Capability: {capability}")
            print(f"  Total Memory: {memory:.2f} GB")
            print(f"  Status: ✅ Usable\n")
        except Exception as e:
            print(f"GPU {i}: ❌ Unavailable - {e}")

if __name__ == "__main__":
    check_gpus()