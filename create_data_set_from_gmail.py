import os.path
import base64
from googleapiclient.discovery import build
from typing import List, Dict, Set
import json
import fcntl

# If modifying scopes, delete token.json
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
EMAILS_BATCH_SIZE = 100

last_page_token = None


class Email:
    # date is in MS since epoch
    def __init__(self, id: str, from_user: str, body: str, date: int):
        self.id = id
        self.from_user = from_user
        self.body = body
        self.date = date

    def to_dict(self):
        return {
            "id": self.id,
            "body": self.body,
            "date": self.date,
            "from_user": self.from_user,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data["id"],
            body=data["body"],
            date=data["date"],
            from_user=data["from_user"],
        )


class EmailThread:
    def __init__(self, thread_id: str, emails: List[Email]):
        self.thread_id = thread_id
        self.emails = sorted(emails, key=lambda x: x.date)

    def add_email(self, email: Email):
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


def get_message_body(message):
    """Extract the body text from a Gmail message."""
    body = ""

    if "parts" in message["payload"]:
        for part in message["payload"]["parts"]:
            if part["mimeType"] == "text/plain":
                if "data" in part["body"]:
                    body = base64.urlsafe_b64decode(part["body"]["data"]).decode(
                        "utf-8"
                    )
                    break
            elif part["mimeType"] == "text/html" and not body:
                if "data" in part["body"]:
                    body = base64.urlsafe_b64decode(part["body"]["data"]).decode(
                        "utf-8"
                    )
    else:
        if message["payload"]["mimeType"] == "text/plain":
            if "data" in message["payload"]["body"]:
                body = base64.urlsafe_b64decode(
                    message["payload"]["body"]["data"]
                ).decode("utf-8")
        elif message["payload"]["mimeType"] == "text/html":
            if "data" in message["payload"]["body"]:
                body = base64.urlsafe_b64decode(
                    message["payload"]["body"]["data"]
                ).decode("utf-8")

    return body


def get_message_date(message):
    """Extract the date from a Gmail message in milliseconds since epoch."""
    return int(message["internalDate"])


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
            for msg in messages:
                try:

                    if msg["id"] in seen_email_ids:
                        if (
                            pagination_token_that_was_loaded_from_file
                            == last_page_token
                        ):
                            # We have this because if we kill the process during the loop, the finally block is run,
                            # which then saves the current state, including the current pagination token in the file.
                            # Now if we rerun the process, it will pick up the current pagination token, and refetch some of the existing email,
                            # which will have emails we may have already seen again. So to prevent the program from stopping in this case,
                            # we do not stop if the current pagination token is the one that was loaded from the file on process start.
                            # On the other hand, if we encounter a message that was have already seen, but the pagination token is not the one
                            # from the file, then we can be sure that we have seen all emails before this one, and we can stop the process (assuming the BATCH SIZE has not been reduced compared to previous runs).
                            continue

                        # this means we have already seen this email before, and therefore all emails
                        # before this one as well (since we are iterating in order from newest to oldest)
                        print(
                            f"Stopping early because we have already seen the past emails before."
                        )
                        stop_early = True
                        break

                    seen_email_ids.add(msg["id"])

                    # Get full message details
                    message = (
                        service.users()
                        .messages()
                        .get(userId="me", id=msg["id"])
                        .execute()
                    )

                    # Extract thread ID, body, and date
                    thread_id = message["threadId"]
                    body = get_message_body(message)
                    date = get_message_date(message)
                    from_user = get_message_from(message)

                    # Create Email object
                    email_obj = Email(
                        id=msg["id"], from_user=from_user, body=body, date=date
                    )

                    # Add to thread and sort immediately
                    if thread_id in threads_dict:
                        threads_dict[thread_id].add_email(email_obj)
                    else:
                        threads_dict[thread_id] = EmailThread(
                            thread_id=thread_id, emails=[email_obj]
                        )

                    total_messages_processed += 1

                    if total_messages_processed % 50 == 0:
                        print(
                            f"Processed {total_messages_processed} messages so far..."
                        )

                except Exception as e:
                    print(
                        f"Error processing message {total_messages_processed + 1}: {e}"
                    )
                    continue

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
