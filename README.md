# email-llm-fine-tuning

## 1) Installation process

### Requirements

Need python 3.10 or higher.

### Install dependencies
```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in the required variables.

### Instructions on creating Gmail API credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable Gmail API
4. Create credentials
    - Add `https://www.googleapis.com/auth/gmail.readonly` scope
    - Make sure to select "Desktop app" in the OAuth Client ID section
5. Download credentials.json, and put it in the repo as `gmail_credentials.json` file.
6. Add the email for which you want to fetch emails from as a test user on google cloud console. This way, you do not need to publish the app or ask google to verify it.

## 2) Creating the token

For the scripts to work, we need to first get an access token from google for the email account. This is done via the OAuth flow wherein you will have to manually login. Run the following command to start the flow:

```bash
python fetch_access_token.py
```

This will write a `token.json` file in the repo, which will be used by the other scripts to fetch the emails.

## 3) Fetching emails

To fetch emails, run the following command:

```bash
python fetch_from_gmail.py
```

This will create a file called `fetching_state.json` in the repo, which will contain all the emails fetched so far, organised by thread ID. This is a long running process, and if it crashes, it can be restarted and it will resume from where it left off.

## 4) Creating the data set

To create the data set, run the following command:

```bash
python create_data_set_from_fetched_emails.py
```

- This will create a file called `dataset.jsonl` in the repo from the contents of the `fetching_state.json` file.

## 5) Running the fine-tuning

### Minimum requirements

Minimum requirements (for Llama fine tuning):
- 16GB GPU RAM
- 128GB hard disk space

If you are fine-tuning a model which has gated access, you need to first go the hugging face page for that model and accept the terms and conditions. Once you have access to the model, you will have to:
- Clone the model repo (which will ask you for your hugging face username and access token). You can find the clone link on the model page on hugging face -> three dots on the top right -> "Clone repository".
- Then clone the model in this repo, and make sure the MODEL_NAME environment variable points to the cloned repo.

### Setting up the cloud instance
Make sure to setup the right env on the cloud instance to access the GPU.

1. Check that you have a GPU instance:
```
lspci | grep -i nvidia
```

2. Install NVIDIA drivers:
```
sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
```

3. Reboot instance
```
sudo reboot
```

4. Verify the driver
```
nvidia-smi
```

5. Install PyTorch with CUDA support
- Go to https://pytorch.org/get-started/locally/
- Select the right OS, Pip, python, Cuda version (you can see the CUDA version from the previous command)
- Copy the command and run it in the cloud instance

6. Save the dependencies
```
pip freeze > requirements.txt
```

7. Check that python is able to detect the GPU
```
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Running the fine-tuning

Run the fine-tuning with Lora (high mem, better quality)
```
python fine-tune-lora.py
```

Run the fine-tuning with QLoRA (low mem, little less quality)
```
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python fine-tune-qlora.py
```

## 6) Inference

Run the inference with Lora
```
python inference-lora.py
```

Run the inference with QLoRA
```
python inference-qlora.py
```
