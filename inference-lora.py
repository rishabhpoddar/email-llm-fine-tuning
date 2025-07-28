import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# Path where your LoRA adapter is saved
peft_model_path = "model_result"

# Load LoRA config and base model
config = PeftConfig.from_pretrained(peft_model_path)

base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",          # puts model on GPU automatically if available
    trust_remote_code=True,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Apply the LoRA adapter
model = PeftModel.from_pretrained(base_model, peft_model_path)
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
