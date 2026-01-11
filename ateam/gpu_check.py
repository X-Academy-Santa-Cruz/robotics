
import torch
import sys

print(f"Python version: {sys.version}")
print(f"Torch version: {torch.__version__}")
print(f"Torch file: {torch.__file__}")

try:
    import intel_extension_for_pytorch as ipex
    print(f"IPEX version: {ipex.__version__}")
except ImportError: 	
    print("IPEX not installed")

print(f"XPU available: {getattr(torch, 'xpu', None) and torch.xpu.is_available()}")
print(f"CUDA available: {torch.cuda.is_available()}")
