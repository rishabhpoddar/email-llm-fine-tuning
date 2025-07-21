import os.path
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# If modifying scopes, delete token.json
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def main():
    creds = None
    # token.json stores the user's access and refresh tokens
    if os.path.exists("token.json"):
        from google.oauth2.credentials import Credentials

        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    else:
        raise Exception("Token file not found. Please run fetch_access_token.py first.")

    # Access Gmail API
    service = build("gmail", "v1", credentials=creds)
    results = service.users().messages().list(userId="me", maxResults=10).execute()
    messages = results.get("messages", [])

    for msg in messages:
        message = service.users().messages().get(userId="me", id=msg["id"]).execute()
        print(f"Snippet: {message['snippet']}")


if __name__ == "__main__":
    main()
