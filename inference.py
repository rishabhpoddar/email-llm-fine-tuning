import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

peft_model_path = "model_result"  # Path where LoRA adapter is saved

# ---- Load base model in 4-bit (same as training) ----
config = PeftConfig.from_pretrained(peft_model_path)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,  # or bfloat16 if supported
)

base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    quantization_config=bnb_config,
    device_map="auto",      # automatically puts on GPU if available
    trust_remote_code=True,
)

# ---- Load tokenizer and LoRA adapter ----
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(base_model, peft_model_path)

# Keep the model on the same device (GPU if available)
model.eval()

# ---- Inference input ----
prompt = "<s>[EMAIL THREAD]\nFrom: foo@bar.com\nSubject: Test\nDate: 2025-07-24 00:00 UTC\n\nHey, can we reschedule our meeting?\n\n[YOUR REPLY]\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# ---- Generate ----
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
