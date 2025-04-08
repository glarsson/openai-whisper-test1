Demo: https://www.youtube.com/watch?v=Qnrrp4Re90E&t=4s

You're gonna need to download and install CUDA 11.8, and cuDNN v8.1.0 (January 26th, 2021), for CUDA 11.0,11.1 and 11.2 for this to work. Extract the cudnn zip into: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
