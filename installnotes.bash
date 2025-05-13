
conda init
conda create --name prompt-guard python=3.12 -c conda-forge
conda activate prompt-guard

#cuda
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

#or cpu
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -U "huggingface_hub[cli]"
# install transformers
pip install -U transformers
pip install -U tiktoken

huggingface-cli login
