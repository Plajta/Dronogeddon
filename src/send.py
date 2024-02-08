import smtplib


from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

class EmailSender:
    def __init__(self,sender,reciver):
        self.sender = sender
        self.reciver = reciver
        self.text = ""

    def create_mail(self,subject,body,img_source):

        message = MIMEMultipart()
        message["From"] = self.sender
        message["To"] = self.reciver
        message["Subject"] = subject
        message["Bcc"] = self.reciver

        message.attach(MIMEText(body, "plain"))

        self.text = MIMEText('<img src="cid:image1">', 'html')
        message.attach(self.text)

        image = MIMEImage(open(img_source, 'rb').read())

        image.add_header('Content-ID', '<image1>')
        message.attach(image)

        self.text = message.as_string()


    def send_mail(self):


        server = smtplib.SMTP('smtp.office365.com', 587)
        server.starttls()
        server.login(self.sender, "Plajta123")
        server.sendmail(self.sender, self.reciver, self.text)
        server.quit()

    def send_intruder_alert(self,img_src):
        subject = "Pozor!! Pozor!!"
        body = "Na vědomost se dává že se tu někdo potuluje.\nToto je ten býdník:"

        self.create_mail(subject,body,img_src)
        self.send_mail()

if __name__ == "__main__":

    sender = "plajta.corporation@hotmail.com"
    reciver = "PlajtaCorp@proton.me"

    mymail = EmailSender(sender,reciver)

    subject = "Pozor!! Pozor!!"
    body = "Na vědomost se dává že se tu někdo potuluje.\nToto je ten býdník:"
    img_src = "src/Flight_logs/img/img.jpg"

    mymail.create_mail(subject,body,img_src)
    mymail.send_mail()