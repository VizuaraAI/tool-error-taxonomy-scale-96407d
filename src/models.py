"""The scale ladder: one thin wrapper over HuggingFace chat models.

Keeping every model behind one interface is what makes the comparison fair.
The prompt template, the generation settings and the tool serialisation are
identical at every scale; only the checkpoint changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

#: The controlled ladder. Same family, same post-training recipe, so parameter
#: count is the only variable that moves.
LADDER: dict[str, str] = {
    "0.5B": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "3B": "Qwen/Qwen2.5-3B-Instruct",
}

#: Cross-family replication: if a 1.7B model trained on a very different data
#: mixture shows the 1.5B error profile, the effect is about scale. If it shows
#: the 0.5B profile, the effect is about tool-use post-training.
CROSS_FAMILY: dict[str, str] = {
    "llama-1B": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3B": "meta-llama/Llama-3.2-3B-Instruct",
    "smol-360M": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "smol-1.7B": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}

#: Approximate parameter counts in billions, used for the elasticity fit.
PARAM_COUNTS: dict[str, float] = {
    "0.5B": 0.5, "1.5B": 1.5, "3B": 3.0,
    "llama-1B": 1.24, "llama-3B": 3.21,
    "smol-360M": 0.36, "smol-1.7B": 1.71,
}

SYSTEM_PROMPT = (
    "You are a tool-calling assistant. You are given a set of tools in JSON "
    "Schema form. Decide which tools to call and with which arguments.\n"
    "Reply with nothing but a JSON list of calls, each of the form "
    '{"name": "<tool name>", "arguments": {<argument name>: <value>}}.\n'
    "Call every tool the request needs, and no tool it does not need. If no "
    "tool applies, reply with an empty list [].")


@dataclass
class GenerationConfig:
    """Generation settings. Greedy is the primary condition for the study."""

    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0

    @property
    def do_sample(self) -> bool:
        return self.temperature > 0.0


def resolve(name: str) -> str:
    """Map a ladder key (``"0.5B"``) or a raw Hub id to a Hub id."""
    return LADDER.get(name) or CROSS_FAMILY.get(name) or name


def build_prompt(query: str, tools: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """Chat messages for one instance. Identical at every scale, by design."""
    import json  # noqa: PLC0415

    rendered = json.dumps(list(tools), indent=2, ensure_ascii=False)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Available tools:\n{rendered}\n\nRequest: {query}"},
    ]


class ChatModel:
    """Lazy-loading wrapper so importing this module needs no GPU or weights."""

    def __init__(self, name: str, device: str = "auto", dtype: str = "auto") -> None:
        self.name = name
        self.model_id = resolve(name)
        self.device = device
        self.dtype = dtype
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        dtype = torch.bfloat16 if self.dtype == "auto" else getattr(torch, self.dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=dtype, device_map=self.device
        )
        self._model.eval()

    def generate(
        self,
        prompts: Sequence[list[dict[str, str]]],
        config: GenerationConfig | None = None,
    ) -> list[str]:
        """Generate one completion per prompt. Returns only the new tokens."""
        import torch  # noqa: PLC0415

        self._ensure_loaded()
        config = config or GenerationConfig()
        torch.manual_seed(config.seed)
        tokenizer, model = self._tokenizer, self._model

        texts = [
            tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        batch = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **batch,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.do_sample,
                temperature=config.temperature if config.do_sample else None,
                top_p=config.top_p if config.do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
            )
        # Strip the prompt: every row was left-padded to the same length.
        new_tokens = output[:, batch["input_ids"].shape[1] :]
        return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
