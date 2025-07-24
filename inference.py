import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# Load LoRA config and base model
peft_model_path = "model_result"  # path where you saved the adapter

# Load base model config from LoRA
config = PeftConfig.from_pretrained(peft_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path, torch_dtype=torch.float32, device_map="auto"
)

# Load tokenizer and apply LoRA adapter
tokenizer = AutoTokenizer.from_pretrained(peft_model_path)
model = PeftModel.from_pretrained(base_model, peft_model_path)

# Move model to CPU if necessary
model.to("cpu")
model.eval()

# ---- Inference input ----
prompt = "<s>[EMAIL THREAD]\nFrom: foo@bar.com\nSubject: Test\nDate: 2025-07-24 00:00 UTC\n\nHey, can we reschedule our meeting?\n\n[YOUR REPLY]\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

# ---- Generate ----
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

# ---- Decode ----
print(tokenizer.decode(output[0], skip_special_tokens=True))
