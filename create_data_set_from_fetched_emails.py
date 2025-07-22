#!/usr/bin/env python3
"""
Convert an email thread dataset to jsonl for LoRA fine-tuning.

Raw format (one big dict):
{
  "<thread_id>": {
      "thread_id": "<thread_id>",
      "emails": [
          {
              "id": "...",
              "body": "...",          # may contain HTML
              "date": 1700000000,     # ms since epoch  (or s — script handles both)
              "from_user": "alice@example.com",
              "subject": "…"
          },
          ...
      ]
  },
  ...
}

For each reply *written by TARGET_ADDRESS* the script emits:
{
  "input":  "<concatenated previous emails>",
  "output": "<target's reply email>"
}
"""


import json
import os
from datetime import datetime, timezone
import html2text
import dotenv

dotenv.load_dotenv()


def strip_html(raw_html: str) -> str:
    """Convert HTML to plain text if html2text is available."""
    if html2text is None:
        return raw_html
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.body_width = 0
    return h.handle(raw_html)


EMAIL_SEPARATOR = "\n\n---\n\n"  # separates individual emails inside the prompt


def serialise_input_email(email: dict) -> str:
    ts = int(email["date"]) / (1000 if email["date"] > 1e12 else 1)  # guess ms vs s
    date_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )

    body = email["body"] or ""
    body = strip_html(body)

    return (
        f"From: {email['from_user']}\n"
        f"Subject: {email.get('subject', '(no subject)')}\n"
        f"Date: {date_str}\n\n"
        f"{body}"
    )


def serialise_output_email(email: dict) -> str:
    body = email["body"] or ""
    body = strip_html(body)
    if os.getenv("EMAIL_ENDING_SIGNATURE_OF_OUTPUT_EMAIL"):
        body = body.split(os.getenv("EMAIL_ENDING_SIGNATURE_OF_OUTPUT_EMAIL"))[0]
    return body


def process_dataset(
    data: dict,
    target_email: str,
):
    """Yield {'input': ..., 'output': ...} pairs."""
    target_email = target_email.lower()

    for thread in data.values():
        # sort e-mails chronologically
        emails = sorted(thread["emails"], key=lambda e: e["date"])

        # build running context as we iterate
        context_blocks = []
        for email in emails:
            # Before appending this email, see if *this* one is by target → create example
            if email["from_user"].lower() == target_email and context_blocks:
                yield {
                    "input": EMAIL_SEPARATOR.join(context_blocks),
                    "output": serialise_output_email(email),
                }

            # Add current email to context for future examples
            context_blocks.append(serialise_input_email(email))


def main():
    # Load raw data
    with open("fetching_state.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Generate pairs and write
    with open("dataset.jsonl", "w", encoding="utf-8") as f_out:
        for pair in process_dataset(
            raw_data["threads_dict"], os.getenv("EMAIL_ADDRESS_TO_TRAIN_ON")
        ):
            f_out.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Wrote dataset.jsonl")


if __name__ == "__main__":
    main()
