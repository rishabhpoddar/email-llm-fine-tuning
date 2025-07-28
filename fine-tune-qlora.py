import os
from typing import Dict

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
import dotenv
from torch.nn.utils.rnn import pad_sequence
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

dotenv.load_dotenv()

if torch.cuda.is_available():
    print("✅ GPU detected:", torch.cuda.get_device_name(0))
else:
    print("⚠️ GPU not available, using CPU")


# ----------  Prompt template ----------
def make_example(example: Dict[str, str], tokenizer):
    # build prompt and reply strings
    prompt = f"<s>[EMAIL THREAD]\n{example['input']}\n\n[YOUR REPLY]\n"
    reply = example["output"] + tokenizer.eos_token

    # tokenize
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    reply_ids = tokenizer(reply, add_special_tokens=False).input_ids

    # concat and build labels
    input_ids = prompt_ids + reply_ids
    labels = [-100] * len(prompt_ids) + reply_ids

    return {"input_ids": input_ids, "labels": labels}


# ----------  Main ----------
def main():
    # Tokeniser
    tokenizer = AutoTokenizer.from_pretrained(
        os.getenv("MODEL_NAME"),
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # QLoRA: load base model in 4-bit (no device_map, single-GPU Trainer will handle device)
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
    # prepare for k-bit training (casts norms, enables input-grad trick)
    model = prepare_model_for_kbit_training(model)

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

    # Dataset -> tokenised
    raw_ds = load_dataset("json", data_files="dataset.jsonl", split="train")

    def _proc(ex):
        return make_example(ex, tokenizer)

    tokenised = raw_ds.map(_proc, batched=False, remove_columns=raw_ds.column_names)

    # Training args
    targs = TrainingArguments(
        output_dir="model_result",
        num_train_epochs=int(os.getenv("EPOCHS")),
        # max_steps=1, # uncomment this and comment the above line for testing
        per_device_train_batch_size=int(os.getenv("BATCH_SIZE")),
        gradient_accumulation_steps=int(os.getenv("GRAD_ACCUM")),
        learning_rate=float(os.getenv("LEARNING_RATE")),
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,         # big memory saver on T4
        optim="paged_adamw_8bit",            # memory-efficient optimizer
        logging_steps=int(os.getenv("TRAINING_LOGGING_STEPS")),
        save_strategy="steps",  # Save at regular step intervals
        save_steps=int(os.getenv("TRAINING_SAVE_STEPS")),
        save_total_limit=int(os.getenv("TRAINING_SAVE_TOTAL_LIMIT")),
        load_best_model_at_end=False,  # Don't need best model for this use case
        resume_from_checkpoint=True,  # Automatically resume if checkpoint exists
        report_to="none",
        label_names=["labels"],
    )

    def collate_fn(batch):
        # batch is a list of dicts: {"input_ids": [...], "labels": [...]}
        input_tensors = [torch.tensor(ex["input_ids"], dtype=torch.long, device="cpu") for ex in batch]
        label_tensors = [torch.tensor(ex["labels"], dtype=torch.long, device="cpu") for ex in batch]

        # pad to the longest in this batch
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

    data_collator = collate_fn

    trainer = Trainer(
        model=model,
        train_dataset=tokenised,
        args=targs,
        data_collator=data_collator,
    )

    # Run the training
    print("Trainer will use device:", trainer.model.device)
    trainer.train()
    trainer.model.save_pretrained("model_result")
    tokenizer.save_pretrained("model_result")
    print("LoRA adapter saved to model_result")


if __name__ == "__main__":
    main()
