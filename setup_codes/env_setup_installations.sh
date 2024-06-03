cd
sudo apt update
sudo apt install -y software-properties-common
sudo apt install ffmpeg
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.9
sudo apt install -y python3.9-venv python3.9-distutils
python3.9 -m ensurepip
python3.9 -m venv py39cuda
source ~/py39cuda/bin/activate
pip install torch deepspeed mosaicml-streaming tqdm click transformers wandb plotly pandas ray
wandb login