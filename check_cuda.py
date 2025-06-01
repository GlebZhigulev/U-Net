import torch

print("CUDA доступна:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Устройство:", torch.cuda.get_device_name(0))
else:
    print("❌ CUDA не работает — скорее всего, установлен PyTorch без поддержки GPU")
