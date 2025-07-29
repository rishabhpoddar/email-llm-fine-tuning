import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with LoRA or QLoRA model"
    )
    parser.add_argument(
        "--method",
        choices=["lora", "qlora"],
        required=True,
        help="Type of fine-tuning (lora or qlora)",
    )
    args = parser.parse_args()

    model_path = f"{args.method}-model"

    # Load LoRA config
    config = PeftConfig.from_pretrained(model_path)

    # Load base model
    if args.method == "qlora":
        # QLoRA: Load with 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        # LoRA: Load normally
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    # Apply the adapter
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    # Prepare input
    prompt = "<s>[EMAIL THREAD]\nFrom: foo@bar.com\nSubject: Test\nDate: 2025-07-24 00:00 UTC\n\nHey, can we reschedule our meeting?\n\n[YOUR REPLY]\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate output
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
