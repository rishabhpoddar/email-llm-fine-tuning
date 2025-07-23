# email-llm-fine-tuning

## 1) Installation process

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

Before we do the fine-tuning, run the following command:

```bash
accelerate config
```

The following are the answers I gave to train on my local machine (mac, m1 pro) with a CPU only:

```
In which compute environment are you running?
This machine
---------------------------------------------------------------------------------------------------------------------------------------
Which type of machine are you using?
No distributed training
Do you want to run your training on CPU only (even if a GPU / Apple Silicon / Ascend NPU device is available)? [yes/NO]:yes
Do you want to use Intel PyTorch Extension (IPEX) to speed up training on CPU/XPU? [yes/NO]:no
Do you wish to optimize your script with torch dynamo?[yes/NO]:no
Do you want to use DeepSpeed? [yes/NO]: no
---------------------------------------------------------------------------------------------------------------------------------------
Do you wish to use mixed precision?
no
```

You can decide to answer them differently, but the `fine-tuning.py` script might need to change a bit.

To run the fine-tuning, run the following command:

```bash
python fine-tune.py
```