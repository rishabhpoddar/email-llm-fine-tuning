"""QLoRA fine-tuning for Llama-4-Scout 17B on e-mail reply data.

Input file: jsonl with
    {"input": "<thread context>", "output": "<author's reply>"}
"""

import os
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
def make_example_batch(examples: Dict[str, list], tokenizer):
    """Process a batch of examples. Return {"input_ids": […], "labels": […]} for the batch."""
    batch_input_ids = []
    batch_labels = []

    for i in range(len(examples["input"])):
        prompt = f"<s>[EMAIL THREAD]\n{examples['input'][i]}\n\n[YOUR REPLY]\n"
        reply = examples["output"][i] + tokenizer.eos_token

        # Tokenise separately to measure length
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        reply_ids = tokenizer(reply, add_special_tokens=False).input_ids

        # here we mask the email thread tokens to -100, so it does not learn to predict those,
        # and instead, just learns from the expected output.
        input_ids = prompt_ids + reply_ids
        labels = [-100] * len(prompt_ids) + reply_ids

        batch_input_ids.append(input_ids)
        batch_labels.append(labels)

    return {"input_ids": batch_input_ids, "labels": batch_labels}


# ----------  Main ----------
def main():
    accelerator = Accelerator()

    # Tokeniser
    tokenizer = AutoTokenizer.from_pretrained(
        os.getenv("MODEL_NAME"), trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # We do this cause otherwise it will require a lot of memory
    qconf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        os.getenv("MODEL_NAME"),
        torch_dtype=torch.float16,
        quantization_config=qconf,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA adapter
    lora_cfg = LoraConfig(
        r=int(os.getenv("LORA_RANK")),
        lora_alpha=int(os.getenv("LORA_ALPHA")),
        lora_dropout=float(os.getenv("LORA_DROPOUT")),
        use_rslora=True,
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

    # Dataset -> tokenised
    raw_ds = load_dataset("json", data_files="dataset.jsonl", split="train")

    def _proc(ex):
        return make_example_batch(ex, tokenizer)

    tokenised = raw_ds.map(
        _proc, batched=True, batch_size=1000, remove_columns=raw_ds.column_names
    )
    # Filter is not needed anymore since we're not returning None values in batched processing

    # Training args
    targs = TrainingArguments(
        output_dir="model_results",
        num_train_epochs=int(os.getenv("EPOCHS")),
        per_device_train_batch_size=int(os.getenv("BATCH_SIZE")),
        gradient_accumulation_steps=int(os.getenv("GRAD_ACCUM")),
        learning_rate=float(os.getenv("LEARNING_RATE")),
        bf16=True,
        logging_steps=25,
        save_strategy="epoch",
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=False, pad_to_multiple_of=8
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenised,
        args=targs,
        data_collator=data_collator,
    )

    # Run the training
    trainer.train()
    accelerator.wait_for_everyone()

    # Save LoRA adapter + tokenizer
    trainer.model.save_pretrained("model_results")
    tokenizer.save_pretrained("model_results")
    if accelerator.is_main_process:
        print(f"🎉  LoRA adapter saved to model_results")


if __name__ == "__main__":
    main()
