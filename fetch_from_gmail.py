import os.path
import base64
from googleapiclient.discovery import build
from typing import List, Dict, Set
import json
import fcntl
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import dotenv
from base64 import urlsafe_b64decode
from email_reply_parser import EmailReplyParser
from bs4 import BeautifulSoup

dotenv.load_dotenv()

# If modifying scopes, delete token.json
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
EMAILS_BATCH_SIZE = int(os.getenv("EMAIL_FETCHING_BATCH_SIZE"))
MAX_WORKERS = int(os.getenv("EMAIL_FETCHING_MAX_PARALLEL_WORKERS"))

last_page_token = None


class Email:
    # date is in MS since epoch
    def __init__(self, id: str, from_user: str, body: str, date: int, subject: str):
        self.id = id
        self.from_user = from_user
        self.body = body
        self.date = date
        self.subject = subject

    def to_dict(self):
        return {
            "id": self.id,
            "body": self.body,
            "date": self.date,
            "from_user": self.from_user,
            "subject": self.subject,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data["id"],
            body=data["body"],
            date=data["date"],
            from_user=data["from_user"],
            subject=data["subject"],
        )


class EmailThread:
    def __init__(self, thread_id: str, emails: List[Email]):
        self.thread_id = thread_id
        self.emails = sorted(emails, key=lambda x: x.date)

    def add_email(self, email: Email):
        # if the email is already in the thread, don't add it
        if email.id in [e.id for e in self.emails]:
            return
        self.emails.append(email)
        self.emails.sort(key=lambda x: x.date)

    def to_dict(self):
        return {
            "thread_id": self.thread_id,
            "emails": [email.to_dict() for email in self.emails],
        }

    @classmethod
    def from_dict(cls, data):
        emails = [Email.from_dict(email_data) for email_data in data["emails"]]
        return cls(thread_id=data["thread_id"], emails=emails)


threads_dict: Dict[str, EmailThread] = {}

seen_email_ids: Set[str] = set()


def _clean_plain(text: str) -> str:
    """Return only the new part of a plain-text message."""
    # library handles signatures & quoting reliably
    return EmailReplyParser.parse_reply(text).strip()


def _clean_html(html: str) -> str:
    """Remove quoted history from a Gmail / generic HTML body."""
    soup = BeautifulSoup(html, "html.parser")

    # common quote containers across clients
    selectors = [
        "blockquote",
        "div.gmail_quote",
        "div.gmail_extra",
        "table.gmail_quote",
        "div.yahoo_quoted",
    ]
    for sel in selectors:
        for node in soup.select(sel):
            node.decompose()

    return soup.get_text("\n", strip=True)


def get_message_body(message):
    """Extract only the author’s fresh text from a Gmail message."""
    for part in message["payload"].get("parts", [message["payload"]]):
        mime = part["mimeType"]
        body_data = part["body"].get("data")
        if not body_data:
            continue

        decoded = urlsafe_b64decode(body_data).decode("utf-8", errors="ignore")

        if mime == "text/plain":
            return _clean_plain(decoded)
        if mime == "text/html":
            return _clean_html(decoded)

    return ""  # fallback – no recognised body part


def get_message_date(message):
    """Extract the date from a Gmail message in milliseconds since epoch."""
    return int(message["internalDate"])


def get_message_subject(message):
    """Extract the subject from a Gmail message."""
    headers = message["payload"]["headers"]
    for header in headers:
        if header["name"].lower() == "subject":
            return header["value"]
    return "unknown"


def get_message_from(message):
    """Extract the sender's email address from a Gmail message."""
    headers = message["payload"]["headers"]
    for header in headers:
        if header["name"].lower() == "from":
            from_value = header["value"]
            # Extract email from "Name <email@domain.com>" format
            if "<" in from_value and ">" in from_value:
                # Extract email between < and >
                start = from_value.find("<") + 1
                end = from_value.find(">")
                return from_value[start:end]
            else:
                # If no < >, assume the whole value is the email
                return from_value.strip()
    return "unknown@unknown.com"  # Fallback if From header not found


def save_state_to_json_file():
    # we write to a temp file first cause in case the process is killed during file writing,
    # we don't corrupt the main file..
    global threads_dict, last_page_token
    temp_file = "fetching_state.json.tmp"
    with open(temp_file, "w") as f:
        # Use file locking to prevent concurrent writes
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            # Convert threads_dict to serializable format
            serializable_threads = {
                thread_id: thread.to_dict()
                for thread_id, thread in threads_dict.items()
            }
            json.dump(
                {
                    "threads_dict": serializable_threads,
                    "last_page_token": last_page_token,
                },
                f,
            )
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    # Atomic rename operation
    os.replace(temp_file, "fetching_state.json")


def load_state_from_json_file():
    global threads_dict, last_page_token, seen_email_ids
    if not os.path.exists("fetching_state.json"):
        threads_dict = {}
        last_page_token = None
        seen_email_ids = set()
        return
    with open("fetching_state.json", "r") as f:
        try:
            state = json.load(f)
            # Convert serialized threads back to EmailThread objects
            loaded_threads = state.get("threads_dict", {})
            threads_dict = {
                thread_id: EmailThread.from_dict(thread_data)
                for thread_id, thread_data in loaded_threads.items()
            }
            last_page_token = state.get("last_page_token")
            seen_email_ids = set()
            for thread_id in threads_dict:
                for email in threads_dict[thread_id].emails:
                    seen_email_ids.add(email.id)
        except Exception as e:
            print(f"Error loading state from json file: {e}")
            threads_dict = {}
            last_page_token = None
            seen_email_ids = set()


def fetch_single_message(
    creds,
    msg_id,
    seen_email_ids,
    pagination_token_that_was_loaded_from_file,
    last_page_token,
):
    """Fetch a single message and return (Email object, thread_id) tuple or special string values."""
    if (
        msg_id in seen_email_ids
        and pagination_token_that_was_loaded_from_file != last_page_token
    ):
        # We have pagination_token_that_was_loaded_from_file != last_page_token condition because
        # if we did not have it, then we may miss out on certain emails that were not saved in the current
        # page from a previous run. For example, if in the previous run, pagination token was p1, and we saved
        # 5 out of the 100 emails for that page, then in the next run, if we do not have this condition, we may
        # quit before saving the remaining 95 emails. Now, having this condition means that we will not stop
        # for the current page, as long as the pagination token is a different one from the one that was last
        # used from the previous run. So this way, in the next run, after we paginate through all 100 emails of the
        # current page, and then on the next page, we encounter an email that we have already seen, then we will
        # can be sure that we have seen all emails in and before that page (assuming the BATCH SIZE has not been reduced compared to previous runs).
        return "stop_early"  # Special return value to indicate continue

    seen_email_ids.add(msg_id)

    # Create a separate Gmail service instance for this thread to avoid SSL conflicts
    service = build("gmail", "v1", credentials=creds)

    # Get full message details
    try:
        message = service.users().messages().get(userId="me", id=msg_id).execute()
        # Extract thread ID, body, and date
        thread_id = message["threadId"]
        body = get_message_body(message)
        date = get_message_date(message)
        from_user = get_message_from(message)
        subject = get_message_subject(message)
        # Create Email object
        email_obj = Email(
            id=msg_id,
            from_user=from_user,
            body=body,
            date=date,
            subject=subject,
        )
        return (email_obj, thread_id)
    except Exception as e:
        print(f"Error getting message {msg_id}: {e}")
        return None


def main():
    global threads_dict, seen_email_ids, last_page_token, EMAILS_BATCH_SIZE
    creds = None
    # token.json stores the user's access and refresh tokens
    if os.path.exists("token.json"):
        from google.oauth2.credentials import Credentials

        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    else:
        raise Exception("Token file not found. Please run fetch_access_token.py first.")

    # Access Gmail API
    service = build("gmail", "v1", credentials=creds)

    print("Fetching all emails from inbox...")

    total_messages_processed = 0

    pagination_token_that_was_loaded_from_file = last_page_token

    # Fetch and process messages using pagination
    stop_early = False
    while True:
        try:
            # Fetch a batch of messages
            if last_page_token:
                results = (
                    service.users()
                    .messages()
                    .list(
                        userId="me",
                        pageToken=last_page_token,
                        maxResults=EMAILS_BATCH_SIZE,
                    )
                    .execute()
                )
            else:
                results = (
                    service.users()
                    .messages()
                    .list(userId="me", maxResults=EMAILS_BATCH_SIZE)
                    .execute()
                )

            messages = results.get("messages", [])
            if not messages:
                break

            # Process each message in this batch
            email_thread_pairs = []  # List of (email_obj, thread_id) tuples

            # Create a list of message IDs to process
            message_ids = [msg["id"] for msg in messages]

            # Use ThreadPoolExecutor to fetch messages in parallel
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit all fetch tasks
                future_to_msg_id = {
                    executor.submit(
                        fetch_single_message,
                        creds,  # Pass creds to the function
                        msg_id,
                        seen_email_ids,
                        pagination_token_that_was_loaded_from_file,
                        last_page_token,
                    ): msg_id
                    for msg_id in message_ids
                }

                # Process completed futures
                for future in as_completed(future_to_msg_id):
                    result = future.result()
                    if result == "stop_early":
                        print(
                            f"Stopping early because we have already seen the past emails before."
                        )
                        stop_early = True
                        break
                    elif result is not None:  # Valid (Email object, thread_id) tuple
                        email_obj, thread_id = result
                        email_thread_pairs.append((email_obj, thread_id))

            # Process all successfully fetched emails
            for email_obj, thread_id in email_thread_pairs:
                # Add to thread
                if thread_id in threads_dict:
                    threads_dict[thread_id].add_email(email_obj)
                else:
                    threads_dict[thread_id] = EmailThread(
                        thread_id=thread_id, emails=[email_obj]
                    )

                total_messages_processed += 1

            print(f"Processed {total_messages_processed} messages so far...")

            if stop_early:
                break

            last_page_token = results.get("nextPageToken")

            if not last_page_token:
                break
        finally:
            save_state_to_json_file()

    print(f"\nProcessing complete!")


if __name__ == "__main__":
    load_state_from_json_file()
    main()
