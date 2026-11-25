# Cache Directory

This directory is reserved for optional local runtime caches created by tools
or development workflows. It is currently unused and holds only `.gitkeep` and this file.

The production dataset and FAISS artifacts are stored in `data/`, not here:

- `data/guardgpt_dataset.jsonl`
- `data/guardgpt_faiss.index`
- `data/guardgpt_id_map.json`

`DatasetLoader` validates and uses those supplied artifacts directly. Do not
move or replace them with files in this directory. Any future cache files are
local runtime artifacts and must not be treated as authoritative dataset or
index sources.

Model caches live elsewhere: the embedding model `all-MiniLM-L6-v2` and the Jev-Style
base model for the fine-tuned JEV are read from the HuggingFace cache in your user
folder (or from the folders set in `GUARDGPT_EMBEDDING_MODEL` and `JEV_BASE_MODEL_DIR`),
not from this directory.
