from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import smtplib

class ServerEmail():
    def __init__(self, mail_host, mail_sender, mail_license, mail_receivers, server_name):
        self.mail_host = str(mail_host)
        self.mail_sender = str(mail_sender)
        self.mail_license = str(mail_license)
        self.mail_receivers = str(mail_receivers)
        self.server_name = str(server_name)

    
    def send_begin_email(self, exp_name, arglist):
        mm = MIMEMultipart('related')
        subject_content = """Begin exp: """ + str(exp_name)
        mm["To"] = "<" + self.mail_receivers + ">"
        mm["From"] = self.server_name + "<" + self.mail_sender + ">"
        mm["Subject"] = Header(subject_content,'utf-8')
        body_content = """Begin To Run Experiment """ + str(arglist)
        message_text = MIMEText(body_content,"plain","utf-8")
        mm.attach(message_text)
        stp = smtplib.SMTP_SSL(self.mail_host)
        stp.connect(self.mail_host, 465)
        stp.login(self.mail_sender, self.mail_license)
        stp.sendmail(self.mail_sender, self.mail_receivers, mm.as_string())
        stp.quit()
        return "Finish Send Begin Email"

    def send_end_email(self, exp_name, arglist):
        mm = MIMEMultipart('related')
        subject_content = """End exp: """ + str(exp_name)
        mm["To"] = "<" + self.mail_receivers + ">"
        mm["From"] = self.server_name + "<" + self.mail_sender + ">"
        mm["Subject"] = Header(subject_content,'utf-8')
        body_content = """Finish Experiment """ + str(arglist)
        message_text = MIMEText(body_content,"plain","utf-8")
        mm.attach(message_text)
        stp = smtplib.SMTP_SSL(self.mail_host)
        stp.connect(self.mail_host, 465)
        stp.login(self.mail_sender, self.mail_license)
        stp.sendmail(self.mail_sender, self.mail_receivers, mm.as_string())
        stp.quit()
        return "Finish Send End Email"  

    def send_einfo_email(self, exp_name, einfo):
        mm = MIMEMultipart('related')
        subject_content = """Errors! in exp: """ + str(exp_name)
        mm["To"] = "<" + self.mail_receivers + ">"
        mm["From"] = self.server_name + "<" + self.mail_sender + ">"
        mm["Subject"] = Header(subject_content,'utf-8')
        body_content = str(einfo)
        message_text = MIMEText(body_content,"plain","utf-8")
        mm.attach(message_text)
        stp = smtplib.SMTP_SSL(self.mail_host)
        stp.connect(self.mail_host, 465)
        stp.login(self.mail_sender, self.mail_license)
        stp.sendmail(self.mail_sender, self.mail_receivers, mm.as_string())
        stp.quit()
        return "Finish Send End Email"      


if __name__ == '__main__':
    Emails = ServerEmail(mail_host="smtp.163.com", 
                                mail_sender="******@163.com", 
                                mail_license="ABCDEFGHIJKLMNOP", 
                                mail_receivers="sihong.he@uconn.edu", 
                                server_name="MyServer")
    print("test send begin email")
    Emails.send_begin_email(exp_name="test send begin email")
    print("test send end email")
    Emails.send_end_email(exp_name="test send end email")
    print("test send error email")
    Emails.send_einfo_email(exp_name="test send error email", einfo="float error")
    print("Finish")