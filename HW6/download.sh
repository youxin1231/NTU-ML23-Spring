# Install required dependencies
pip install einops
pip install transformers
pip install ema_pytorch
pip install accelerate
pip install kaggle
pip install stylegan2_pytorch

# Download dataset
if [ ! -d data ]; then
    kaggle datasets download -d b07202024/diffusion
    unzip diffusion.zip && rm diffusion.zip
    mv faces/faces data && rmdir faces
fi