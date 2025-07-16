from __future__ import annotations
from typing import Optional, Dict, Any, List, Tuple
import base64
import mimetypes
import os
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Google API Authentication Libraries
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import AuthorizedSession, Request
from google.auth.exceptions import RefreshError
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES: str = "https://www.googleapis.com/auth/gmail.send"
CLIENT_SECRET_FILE: str = "client_secret.json"
APPLICATION_NAME: str = "Gmail API Python Send Email"


def get_credentials() -> Credentials:
    """
    Get valid user credentials from storage.
    
    If no credentials are available, initiate the OAuth2 authorization flow to obtain new credentials.
    
    Returns:
        Credentials: Valid OAuth2 credentials for accessing Gmail API.
    """
    creds: Optional[Credentials] = None
    home_dir: str = os.path.expanduser("~")
    credential_dir: str = os.path.join(home_dir, ".credentials")
    
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    
    credential_path: str = os.path.join(credential_dir, "gmail-python-email-send.json")
    
    # Attempt to load existing credentials
    if os.path.exists(credential_path):
        try:
            creds = Credentials.from_authorized_user_file(credential_path, SCOPES)
        except ValueError as e:
            print(f"Error loading credentials: {e}")
    
    # Refresh expired credentials or obtain new ones
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())  # Use Request from google.auth.transport.requests
            except RefreshError:
                print("Failed to refresh credentials. Re-authorizing...")
                creds = None
        if not creds:
            flow: InstalledAppFlow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRET_FILE, SCOPES
            )
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for future use
        with open(credential_path, 'w') as token:
            token.write(creds.to_json())
        print(f"Storing credentials to {credential_path}")
    
    return creds


def SendMessage(
    sender: str, 
    to: str, 
    subject: str, 
    msgHtml: str, 
    msgPlain: str, 
    attachmentFile: Optional[str] = None
) -> Dict[str, Any] | str:
    """
    Send an email message via the Gmail API.
    
    Args:
        sender: Email address of the sender.
        to: Email address of the recipient.
        subject: The subject of the email message.
        msgHtml: HTML content of the email.
        msgPlain: Plain text version of the email.
        attachmentFile: Path to the file to be attached (optional).
    
    Returns:
        Dict[str, Any]: The sent message details if successful.
        str: Error message if an error occurred.
    """
    credentials: Credentials = get_credentials()
    
    try:
        # Build the Gmail API service
        service = build("gmail", "v1", credentials=credentials)
        
        if attachmentFile:
            message = createMessageWithAttachment(
                sender, to, subject, msgHtml, msgPlain, attachmentFile
            )
        else:
            message = CreateMessageHtml(sender, to, subject, msgHtml, msgPlain)
        
        result = SendMessageInternal(service, "me", message)
        return result
    
    except HttpError as error:
        print(f"An error occurred: {error}")
        return "Error"


def SendMessageInternal(
    service: Any, 
    user_id: str, 
    message: Dict[str, str]
) -> Dict[str, Any] | str:
    """
    Internal helper function to send an email message.
    
    Args:
        service: Authorized Gmail API service instance.
        user_id: User's email address. Use "me" for the authenticated user.
        message: Message to be sent.
    
    Returns:
        Dict[str, Any]: Sent message details if successful.
        str: Error message if an error occurred.
    """
    try:
        message_response = (
            service.users().messages().send(userId=user_id, body=message).execute()
        )
        print(f"Message Id: {message_response['id']}")
        return message_response
    except HttpError as error:
        print(f"An error occurred: {error}")
        return "Error"
    return "OK"


def createMessageWithAttachment(
    sender: str, 
    to: str, 
    subject: str, 
    msgHtml: str, 
    msgPlain: str, 
    attachmentFile: str
) -> Dict[str, str]:
    """Create a MIME message with an attachment.
    
    Args:
        sender: Email address of the sender.
        to: Email address of the receiver.
        subject: The subject of the email message.
        msgHtml: HTML content of the email.
        msgPlain: Plain text version of the email.
        attachmentFile: Path to the file to be attached.
    
    Returns:
        Dict[str, str]: A dictionary containing the base64url encoded email object.
    """
    message = MIMEMultipart("mixed")
    message["to"] = to
    message["from"] = sender
    message["subject"] = subject

    messageA = MIMEMultipart("alternative")
    messageR = MIMEMultipart("related")

    messageR.attach(MIMEText(msgHtml, "html"))
    messageA.attach(MIMEText(msgPlain, "plain"))
    messageA.attach(messageR)

    message.attach(messageA)

    print(f"Creating message with attachment: {attachmentFile}")
    content_type, encoding = mimetypes.guess_type(attachmentFile)

    if content_type is None or encoding is not None:
        content_type = "application/octet-stream"
    
    main_type, sub_type = content_type.split("/", 1)
    
    with open(attachmentFile, "rb") as fp:
        if main_type == "text":
            msg = MIMEText(fp.read().decode("utf-8"), _subtype=sub_type)
        elif main_type == "image":
            msg = MIMEImage(fp.read(), _subtype=sub_type)
        elif main_type == "audio":
            msg = MIMEAudio(fp.read(), _subtype=sub_type)
        else:
            msg = MIMEBase(main_type, sub_type)
            msg.set_payload(fp.read())
    
    filename = os.path.basename(attachmentFile)
    msg.add_header("Content-Disposition", "attachment", filename=filename)
    message.attach(msg)

    return {"raw": base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")}


def CreateMessageHtml(
    sender: str, 
    to: str, 
    subject: str, 
    msgHtml: str, 
    msgPlain: str
) -> Dict[str, str]:
    """Create a MIME message without attachments.
    
    Args:
        sender: Email address of the sender.
        to: Email address of the receiver.
        subject: The subject of the email message.
        msgHtml: HTML content of the email.
        msgPlain: Plain text version of the email.
    
    Returns:
        Dict[str, str]: A dictionary containing the base64url encoded email object.
    """
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to
    msg.attach(MIMEText(msgPlain, "plain"))
    msg.attach(MIMEText(msgHtml, "html"))
    
    return {"raw": base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")}


def main() -> None:
    """Main function to send an email interactively."""
    to = input("Enter recipient email address: ")
    sender = input("Your email address: ")
    subject = input("Enter subject: ")
    msgHtml = input("Enter HTML message: ")
    msgPlain = "Hi\nThis is the plain text version of the email."
    
    SendMessage(sender, to, subject, msgHtml, msgPlain)
    # Example of sending with attachment:
    # SendMessage(sender, to, subject, msgHtml, msgPlain, '/path/to/file.pdf')


if __name__ == "__main__":
    main()