import httplib2
import os
import oauth2client
from oauth2client import client, tools
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from apiclient import errors, discovery
import mimetypes
from email.mime.image import MIMEImage
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase

SCOPES = 'https://www.googleapis.com/auth/gmail.send'
CLIENT_SECRET_FILE = 'client_secret.json'
APPLICATION_NAME = 'Gmail API Python Send Email'

def get_credentials():
    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(credential_dir,
                                   'gmail-python-email-send.json')
    store = oauth2client.file.Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = APPLICATION_NAME
        credentials = tools.run_flow(flow, store)
        print 'Storing credentials to ' + credential_path
    return credentials

def SendMessage(sender, to, subject, msgHtml, msgPlain, attachmentFile=None):
    credentials = get_credentials()
    http = credentials.authorize(httplib2.Http())
    service = discovery.build('gmail', 'v1', http=http)
    if attachmentFile:
        message1 = createMessageWithAttachment(sender, to, subject, msgHtml, msgPlain, attachmentFile)
    else: 
        message1 = CreateMessageHtml(sender, to, subject, msgHtml, msgPlain)
    result = SendMessageInternal(service, "me", message1)
    return result

def SendMessageInternal(service, user_id, message):
    try:
        message = (service.users().messages().send(userId=user_id, body=message).execute())
        print 'Message Id: %s' % message['id']
        return message
    except errors.HttpError, error:
        print 'An error occurred: %s' % error
        return "Error"
    return "OK"

#Create Mail With AttAchment Remove hash to make the code readable
#def createMessageWithAttachment(
#    sender, to, subject, msgHtml, msgPlain, attachmentFile):
    """Create a message for an email.

    Args:
      sender: Email address of the sender.
      to: Email address of the receiver.
      subject: The subject of the email message.
      msgHtml: Html message to be sent
      msgPlain: Alternative plain text message for older email clients          
      attachmentFile: The path to the file to be attached.

    Returns:
      An object containing a base64url encoded email object.
    """
#    message = MIMEMultipart('mixed')
#    message['to'] = to
#    message['from'] = sender
#    message['subject'] = subject

#    messageA = MIMEMultipart('alternative')
#    messageR = MIMEMultipart('related')

#    messageR.attach(MIMEText(msgHtml, 'html'))
#    messageA.attach(MIMEText(msgPlain, 'plain'))
#    messageA.attach(messageR)

#    message.attach(messageA)

#    print "create_message_with_attachment: file:", attachmentFile
#    content_type, encoding = mimetypes.guess_type(attachmentFile)

#    if content_type is None or encoding is not None:
#        content_type = 'application/octet-stream'
#    main_type, sub_type = content_type.split('/', 1)
#    if main_type == 'text':
#        fp = open(attachmentFile, 'rb')
#        msg = MIMEText(fp.read(), _subtype=sub_type)
#        fp.close()
#    elif main_type == 'image':
#        fp = open(attachmentFile, 'rb')
#        msg = MIMEImage(fp.read(), _subtype=sub_type)
#        fp.close()
#    elif main_type == 'audio':
#        fp = open(attachmentFile, 'rb')
#        msg = MIMEAudio(fp.read(), _subtype=sub_type)
#        fp.close()
#    else:
#        fp = open(attachmentFile, 'rb')
#        msg = MIMEBase(main_type, sub_type)
#        msg.set_payload(fp.read())
#        fp.close()
#    filename = os.path.basename(attachmentFile)
#    msg.add_header('Content-Disposition', 'attachment', filename=filename)
#    message.attach(msg)

#    return {'raw': base64.urlsafe_b64encode(message.as_string())}


def CreateMessageHtml(sender, to, subject, msgHtml, msgPlain):
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = to
    msg.attach(MIMEText(msgPlain, 'plain'))
    msg.attach(MIMEText(msgHtml, 'html'))
    return {'raw': base64.urlsafe_b64encode(msg.as_string())}
def main():
    to = input("Enter Email Address: ")
    sender = input("Your Mail ID: ")
    subject = input("Enter your Subject: ")
    msgHtml = input("Enter your Message: ")
    msgPlain = "Hi\nPlain Email"
    SendMessage(sender, to, subject, msgHtml, msgPlain)
    # Send message with attachment: 
    #SendMessage(sender, to, subject, msgHtml, msgPlain, '/path/to/file.pdf')

if __name__ == '__main__':
    main()
