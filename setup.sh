#!/usr/bin/env bash
# GuardGPT setup (Linux / macOS). Run from the project root:  bash setup.sh
set -euo pipefail

[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

[ -f .env ] || { cp .env.example .env; echo "Created .env from .env.example"; }

# The embedding model is loaded offline by default; cache it once.
GUARDGPT_ALLOW_MODEL_DOWNLOAD=1 python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('Embedding model cached.')"

if command -v ollama >/dev/null 2>&1; then
  ollama pull llama3.2
else
  echo "WARNING: Ollama not found. Install from https://ollama.com, then run: ollama pull llama3.2"
fi

echo
echo "Done. Next:"
echo "  source .venv/bin/activate"
echo "  python verify_augmented_dataset.py"
echo "  python main.py --status"
echo "  python main.py --chat"
