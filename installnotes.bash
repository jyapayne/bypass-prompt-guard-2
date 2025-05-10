
conda init
conda create --name prompt-guard python=3.12 -c conda-forge
conda activate prompt-guard

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup install 1.82
rustup default 1.82

pip install fschat
#cuda
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

#or cpu
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -U "huggingface_hub[cli]"
# install transformers
pip install transformers
export RUSTFLAGS="-A invalid_reference_casting"
pip install git+https://github.com/llm-attacks/llm-attacks

huggingface-cli login
