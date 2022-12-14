import smtplib, ssl
import yaml
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def sendEmail():
    with open("secret.yml", "r") as stream:
        try:
            data_sec = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    sender_email = "plajtacorp@gmail.com"
    password = data_sec["password"]
    filename = "imgs/img.jpg"

    with open("sender.yml", "r") as stream:
        try:
            data_pub = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    receiver_email = data_pub["email"]
    name = data_pub["name"]

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "! ACHTUNG ! " + name
    message.attach(MIMEText("Narušení koridoru!", "plain"))

    with open(filename, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    encoders.encode_base64(part)

    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {filename}",
    )

    message.attach(part)
    text = message.as_string()

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, text)

    print("sent")