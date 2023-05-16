# Package installation
# Training progress bar
pip install -q qqdm

# Downloading data

if [ ! -d data ]; then
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh |  bash
    sudo apt-get install -y --allow-unauthenticated git-lfs
    git clone https://github.com/chiyuanhsiao/ml2023spring-hw8
    cd ml2023spring-hw8
    git lfs install
    git lfs pull
    mv ml2023spring-hw8 data
fi