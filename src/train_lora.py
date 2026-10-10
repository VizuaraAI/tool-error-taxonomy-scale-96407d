"""Optional week 7 extension: a small LoRA fine-tune on tool-call data.

The main study is an evaluation study and needs no training. This entry point
exists for one specific question: are the error types that scale does NOT fix
(high elasticity alpha, near zero) also the ones a light fine-tune does not
fix? If a LoRA on 2k synthetic calls removes E3 and E4 at 0.5B but leaves E1
untouched, that is strong evidence the semantic errors are capacity-limited
rather than format-limited.

    python -m src.train_lora --model 0.5B --data data/toolcalls.jsonl --epochs 2

Expects a .jsonl with {"messages": [...]} per line, in the chat format the
tokenizer's template understands.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from .models import resolve

DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
]


def load_chat_jsonl(path: str | pathlib.Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"{path} is empty")
    if "messages" not in rows[0]:
        raise ValueError("each line needs a 'messages' field in chat format")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="0.5B")
    parser.add_argument("--data", required=True, help="chat-format .jsonl")
    parser.add_argument("--out", default="checkpoints/lora")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args(argv)

    # Heavy imports stay inside main so `python -m src.train_lora --help` works
    # on a laptop with nothing but the standard library installed.
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    model_id = resolve(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto"
    )

    dataset = Dataset.from_list(load_chat_jsonl(args.data))
    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=DEFAULT_TARGET_MODULES,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=SFTConfig(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            max_length=args.max_length,
            logging_steps=10,
            save_strategy="epoch",
            bf16=True,
            report_to=[],
        ),
    )
    trainer.train()
    trainer.save_model(args.out)
    print(f"adapter saved to {args.out}")
    print("now evaluate it: python -m src.evaluate --model", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
