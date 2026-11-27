# Data Directory

This directory contains the supplied GuardGPT dataset and its validated vector
search artifacts:

- `guardgpt_dataset.jsonl` - 19,200 JSONL records with `input_text`, `intent`,
	`confidence`, and `reason` fields. The six intent labels are balanced, 3,200
	records each: `safe`, `prompt_injection`, `jailbreak`, `harmful_instructions`,
	`manipulation`, `self_harm_risk`. The `confidence` field ranges from 0.5 to 0.95,
	so the labels are not all certain.
- `guardgpt_faiss.index` - FAISS `IndexFlatIP` containing 19,200 normalized
	384-dimensional vectors.
- `guardgpt_id_map.json` - position-to-record mapping validated against the
	JSONL dataset and FAISS index.
- `guardgpt_id_map.json.bak` - an optional backup copy of the ID map, if you keep one; it
	is not used unless explicitly restored outside the application.

`core.dataset_loader.DatasetLoader` loads the JSONL records, validates the
matching index and ID map, and returns the nearest records with their similarity.
Two layers use them:

- Layer 2 (`IntentIntelligence`) blends the neighbours' labels with the trained head, the
  curated anchors and Layer 1's findings. The dataset is one source in that ensemble
  (default weight 0.2), not a replacement for it.
- Layer 1 treats a near-duplicate of a known unsafe record as extra evidence.

If the files are missing, both layers still run without the dataset evidence, and
`python main.py --status` reports it.

## Antivirus quarantine

Windows Defender has quarantined `guardgpt_dataset.jsonl` and `guardgpt_id_map.json` as
`Exploit:Python/Leivion.A`, several times (25-27 Nov 2026). If they disappear, check
**Windows Security → Virus & threat protection → Protection history** and restore them
from there, from your project archive, or with `git checkout -- data/` (the files are in
the repository). Defender may quarantine them again. Allow the files or exclude this
folder only if you trust where the dataset came from, since that reduces protection.

While a file is missing, `python main.py --status` reports `"dataset": false`, Layer 1
and Layer 2 lose their dataset evidence, and the accuracy test
`test_safe_prompts_are_never_blocked` can fail, because results differ without it.

The index uses the normalized 384-dimensional embeddings from
`all-MiniLM-L6-v2`. Do not edit, regenerate, or reorder these artifacts without
also rebuilding and validating the complete dataset/index/ID-map set. Check them with:

```bash
python verify_augmented_dataset.py
```

## Known data notes

A review of the dataset found it structurally sound (valid labels, no empty texts, index
and id map aligned, none of the project's evaluation prompts present in it). Minor
issues: about 24 duplicate rows, 5 texts that carry two different labels (mostly insults
tagged both `harmful_instructions` and `manipulation`, and `jailbreak` versus
`prompt_injection`), and some rows with garbled text. Any cleanup requires rebuilding
the FAISS index and id map together.

The dataset vote can lean heavily towards `self_harm_risk` on everyday messages. In a test
chat on 2026-11-25, "I am feeling a little bored today" received a dataset vote of 100%
`self_harm_risk`.

## Evaluation prompts

The labelled prompts used to measure accuracy are not in this folder: they are in
`tests/data/intent_samples.json` (119 prompts in three sets: `tuning` 47, `holdout` 37 and
`fresh` 35, which was never used for tuning). None of them is in this dataset, and only
"hello" (in `tuning`) also appears in `intent_classifier/intent_anchors.json`.
