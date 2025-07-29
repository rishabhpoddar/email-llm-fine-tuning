import os
import argparse
from typing import Dict

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import dotenv
from torch.nn.utils.rnn import pad_sequence

dotenv.load_dotenv()

if torch.cuda.is_available():
    print("✅ GPU detected:", torch.cuda.get_device_name(0))
else:
    print("⚠️ GPU not available, using CPU")


def make_example(example: Dict[str, str], tokenizer):
    prompt = f"<s>[EMAIL THREAD]\n{example['input']}\n\n[YOUR REPLY]\n"
    reply = example["output"] + tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    reply_ids = tokenizer(reply, add_special_tokens=False).input_ids

    input_ids = prompt_ids + reply_ids
    labels = [-100] * len(prompt_ids) + reply_ids

    return {"input_ids": input_ids, "labels": labels}


def setup_model_and_tokenizer(use_qlora: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        os.getenv("MODEL_NAME"),
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            os.getenv("MODEL_NAME"),
            quantization_config=bnb_config,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            os.getenv("MODEL_NAME"),
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if torch.cuda.is_available():
            model.to("cuda")

    return model, tokenizer


def setup_lora_config():
    return LoraConfig(
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


def create_collate_fn(tokenizer, use_qlora: bool):
    def collate_fn(batch):
        device = "cpu" if use_qlora else None
        input_tensors = [
            torch.tensor(ex["input_ids"], dtype=torch.long, device=device) if device else torch.tensor(ex["input_ids"], dtype=torch.long)
            for ex in batch
        ]
        label_tensors = [
            torch.tensor(ex["labels"], dtype=torch.long, device=device) if device else torch.tensor(ex["labels"], dtype=torch.long)
            for ex in batch
        ]

        input_ids = pad_sequence(
            input_tensors, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        labels = pad_sequence(label_tensors, batch_first=True, padding_value=-100)
        attention_mask = input_ids.ne(tokenizer.pad_token_id)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    return collate_fn


def main():
    parser = argparse.ArgumentParser(description="Fine-tune model with LoRA or QLoRA")
    parser.add_argument(
        "--method",
        choices=["lora", "qlora"],
        required=True,
        help="Choose fine-tuning method: lora or qlora"
    )
    args = parser.parse_args()

    use_qlora = args.method == "qlora"
    output_dir = f"{args.method}-model"

    model, tokenizer = setup_model_and_tokenizer(use_qlora)

    lora_cfg = setup_lora_config()
    model = get_peft_model(model, lora_cfg)

    raw_ds = load_dataset("json", data_files="dataset.jsonl", split="train")

    def _proc(ex):
        return make_example(ex, tokenizer)

    tokenised = raw_ds.map(_proc, batched=False, remove_columns=raw_ds.column_names)

    training_args_kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": int(os.getenv("EPOCHS")),
        "per_device_train_batch_size": int(os.getenv("BATCH_SIZE")),
        "gradient_accumulation_steps": int(os.getenv("GRAD_ACCUM")),
        "learning_rate": float(os.getenv("LEARNING_RATE")),
        "fp16": torch.cuda.is_available(),
        "logging_steps": int(os.getenv("TRAINING_LOGGING_STEPS")),
        "save_strategy": "steps",
        "save_steps": int(os.getenv("TRAINING_SAVE_STEPS")),
        "save_total_limit": int(os.getenv("TRAINING_SAVE_TOTAL_LIMIT")),
        "load_best_model_at_end": False,
        "resume_from_checkpoint": True,
        "report_to": "none",
        "label_names": ["labels"],
    }

    if use_qlora:
        training_args_kwargs.update({
            "gradient_checkpointing": True,
            "optim": "paged_adamw_8bit",
        })

    targs = TrainingArguments(**training_args_kwargs)

    data_collator = create_collate_fn(tokenizer, use_qlora)

    trainer = Trainer(
        model=model,
        train_dataset=tokenised,
        args=targs,
        data_collator=data_collator,
    )

    print("Trainer will use device:", trainer.model.device)
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"LoRA adapter saved to {output_dir}")


if __name__ == "__main__":
    main()