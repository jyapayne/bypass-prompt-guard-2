# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

uv venv --python 3.12.0

#cuda
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

#or cpu
uv pip install torch --index-url https://download.pytorch.org/whl/cpu

uv pip install -U "huggingface_hub[cli]" transformers tiktoken

huggingface-cli login
