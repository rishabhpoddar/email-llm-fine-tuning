"""QLoRA fine-tuning for Llama-4-Scout 17B on e-mail reply data.

Input file: jsonl with
    {"input": "<thread context>", "output": "<author's reply>"}
"""

import argparse, json, os
from typing import Dict

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from trl import SFTTrainer
from bitsandbytes import BitsAndBytesConfig


# ----------  Prompt template ----------
def make_example(example: Dict[str, str], tokenizer):
    """Return {"input_ids": …, "labels": …} or None if too long."""
    sep = "\n\n---\n\n"
    prompt = f"<s>[EMAIL THREAD]\n{example['input']}\n\n[YOUR REPLY]\n"
    reply = example["output"] + tokenizer.eos_token

    # Tokenise separately to measure length
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    reply_ids = tokenizer(reply, add_special_tokens=False).input_ids

    # Build labels: mask prompt tokens to -100 so loss only on reply
    input_ids = prompt_ids + reply_ids
    labels = [-100] * len(prompt_ids) + reply_ids
    return {"input_ids": input_ids, "labels": labels}


# ----------  Main ----------
def main():
    accelerator = Accelerator()
    device = accelerator.device

    # 1. Tokeniser
    tok = AutoTokenizer.from_pretrained(
        os.getenv("MODEL_NAME"), use_fast=True, trust_remote_code=True
    )
    tok.pad_token = tok.eos_token

    # 2. Model w/ 4-bit quant
    qconf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        os.getenv("MODEL_NAME"),
        torch_dtype=torch.bfloat16,
        quantization_config=qconf,
        device_map="auto",
        trust_remote_code=True,
    )

    # 3. LoRA adapter
    lora_cfg = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 4. Dataset ↦ tokenised
    raw_ds = load_dataset("json", data_files="dataset.jsonl", split="train")

    def _proc(ex):
        return make_example(ex, tok)

    tokenised = raw_ds.map(_proc, remove_columns=raw_ds.column_names)
    tokenised = tokenised.filter(lambda e: e is not None)  # drop None

    # 5. Training args
    targs = TrainingArguments(
        output_dir="model_results",
        num_train_epochs=os.getenv("EPOCHS"),
        per_device_train_batch_size=os.getenv("BATCH_SIZE"),
        gradient_accumulation_steps=os.getenv("GRAD_ACCUM"),
        learning_rate=os.getenv("LEARNING_RATE"),
        bf16=True,
        logging_steps=25,
        save_strategy="epoch",
        report_to="none",
    )

    # 6. Trainer (TRL’s SFTTrainer hides label-shift boilerplate)
    data_collator = DataCollatorForLanguageModeling(
        tok, mlm=False, pad_to_multiple_of=8
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=tokenised,
        args=targs,
        data_collator=data_collator,
    )

    # 7. Go!
    trainer.train()
    accelerator.wait_for_everyone()

    # 8. Save LoRA adapter + tokenizer
    trainer.model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    if accelerator.is_main_process:
        print(f"🎉  LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
