import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import asyncio
from app.config.settings import settings

logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self):
        self.server = settings.SMTP_SERVER
        self.port = settings.SMTP_PORT
        self.username = settings.SMTP_USERNAME
        self.password = settings.SMTP_PASSWORD
        self.from_email = settings.SMTP_USERNAME if settings.SMTP_USERNAME else "noreply@memoryverse.app"

    def _send_email_sync(self, to_email: str, subject: str, html_body: str) -> bool:
        if not self.username or not self.password:
            logger.error("SMTP credentials are not configured. Email not sent.")
            return False
            
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.from_email
            msg["To"] = to_email

            part = MIMEText(html_body, "html")
            msg.attach(part)

            if self.port == 465:
                # SSL
                with smtplib.SMTP_SSL(self.server, self.port) as server:
                    server.login(self.username, self.password)
                    server.sendmail(self.from_email, to_email, msg.as_string())
            else:
                # TLS
                with smtplib.SMTP(self.server, self.port) as server:
                    server.starttls()
                    server.login(self.username, self.password)
                    server.sendmail(self.from_email, to_email, msg.as_string())
                    
            logger.info(f"Email sent successfully to {to_email}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email to {to_email} via SMTP. Error: {str(e)}")
            return False

    async def _send_email(self, to_email: str, subject: str, html_body: str) -> bool:
        return await asyncio.to_thread(self._send_email_sync, to_email, subject, html_body)

    async def send_verification_email(self, to_email: str, name: str, otp: str, expire_mins: int) -> bool:
        subject = f"Verify your {settings.PROJECT_NAME} account"
        html = f"""
        <div style="font-family: sans-serif; max-width: 600px; margin: 0 auto;">
            <h2>Welcome to {settings.PROJECT_NAME}!</h2>
            <p>Hi {name},</p>
            <p>Your email verification code is:</p>
            <h1 style="letter-spacing: 4px; padding: 10px; background-color: #f4f4f4; text-align: center; border-radius: 5px;">
                {otp}
            </h1>
            <p>This code will expire in {expire_mins} minutes.</p>
            <p style="color: #666; font-size: 0.9em; margin-top: 30px;">
                <strong>Security Warning:</strong> Never share this code with anyone. 
                Our staff will never ask for your code.
            </p>
        </div>
        """
        return await self._send_email(to_email, subject, html)

    async def send_password_reset_email(self, to_email: str, name: str, otp: str, expire_mins: int) -> bool:
        subject = f"Password reset for {settings.PROJECT_NAME}"
        
        greeting = f"Hi {name}," if name else "Hi there,"
        
        html = f"""
        <div style="font-family: sans-serif; max-width: 600px; margin: 0 auto;">
            <h2>Password Reset Request</h2>
            <p>{greeting}</p>
            <p>We received a request to reset your password. Here is your code:</p>
            <h1 style="letter-spacing: 4px; padding: 10px; background-color: #f4f4f4; text-align: center; border-radius: 5px;">
                {otp}
            </h1>
            <p>This code will expire in {expire_mins} minutes.</p>
            <p>If you did not request a password reset, you can safely ignore this email.</p>
            <p style="color: #666; font-size: 0.9em; margin-top: 30px;">
                <strong>Security Warning:</strong> Never share this code with anyone. 
                Our staff will never ask for your code.
            </p>
        </div>
        """
        return await self._send_email(to_email, subject, html)

email_service = EmailService()
