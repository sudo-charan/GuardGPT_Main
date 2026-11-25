"""Example JEV adapter - copy, fill in, and point JEV_ADAPTER at your copy.

GuardGPT calls:

    scorer = load(checkpoint_path, base_model_dir, device)
    scores = scorer.scores(text)   # {"safe": s0, "prompt_injection": s1, ..., "self_harm_risk": s5}

Scores can be raw verdict scores (yes-minus-no); GuardGPT applies softmax and
argmax itself. Use the SAME renderer, tokenizer and verdict-slot logic as
scripts/train_jev.py / scripts/eval_jev_finetuned.py, otherwise the 81.8 %
test accuracy will not carry over.
"""

LABELS = ("safe", "prompt_injection", "jailbreak", "harmful_instructions", "manipulation", "self_harm_risk")


class Scorer:
    def __init__(self, model, tokenizer, device):
        self.model, self.tokenizer, self.device = model, tokenizer, device

    def scores(self, text):
        # TODO: paste the renderer + verdict scoring from eval_jev_finetuned.py, e.g.
        #   rendered, slot_positions = render(text, LABELS)          # your JEV renderer
        #   hidden = self.model(..., output_hidden_states=True).hidden_states[-1][0]
        #   direction = E[yes_id] - E[no_id]
        #   values = hidden[slot_positions] @ direction
        #   return dict(zip(LABELS, values.tolist()))
        raise NotImplementedError("copy the verdict scoring from your eval script")


def load(checkpoint_path, base_model_dir, device="cpu"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    model = AutoModelForCausalLM.from_pretrained(base_model_dir, torch_dtype=torch.bfloat16)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = state.get("model_state_dict", state)
    model.load_state_dict(state, strict=False)   # TODO: same loading code as your eval script
    return Scorer(model.to(device).eval(), tokenizer, device)
