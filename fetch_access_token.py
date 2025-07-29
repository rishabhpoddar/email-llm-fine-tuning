import os.path
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
import dotenv
from google.oauth2.credentials import Credentials

dotenv.load_dotenv()

# If modifying scopes, delete token.json
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def main():
    creds = None
    # token.json stores the user's access and refresh tokens
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "gmail_credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)

        # Save the credentials
        with open("token.json", "w") as token:
            token.write(creds.to_json())


if __name__ == "__main__":
    main()
