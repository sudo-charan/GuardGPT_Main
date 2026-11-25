# GuardGPT setup (Windows / PowerShell). Run from the project root:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\setup.ps1
$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment (.venv)..."
    python -m venv .venv
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example"
}

# The embedding model is loaded offline by default; cache it once.
$env:GUARDGPT_ALLOW_MODEL_DOWNLOAD = "1"
& .\.venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('Embedding model cached.')"

if (Get-Command ollama -ErrorAction SilentlyContinue) {
    ollama pull llama3.2
} else {
    Write-Warning "Ollama not found. Install it from https://ollama.com, then run: ollama pull llama3.2"
}

Write-Host ""
Write-Host "Done. Next:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python verify_augmented_dataset.py"
Write-Host "  python main.py --status"
Write-Host "  python main.py --chat"
