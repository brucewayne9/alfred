"""Multi-account IMAP/SMTP email client."""

import imaplib
import smtplib
import email
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header
from email.utils import parsedate_to_datetime
from datetime import datetime
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv("/home/aialfred/alfred/config/.env")

logger = logging.getLogger(__name__)

# Email account configurations
EMAIL_ACCOUNTS = {
    "groundrush": {
        "email": "mjohnson@groundrushlabs.com",
        "name": "Groundrush Labs",
        "imap_server": "mail.doowoprnb.com",
        "imap_port": 993,
        "smtp_server": "mail.doowoprnb.com",
        "smtp_port": 465,
        "use_ssl": True,
        "password_env": "EMAIL_PASS_GROUNDRUSH",
    },
    "rucktalk": {
        "email": "info@rucktalk.com",
        "name": "Ruck Talk",
        "imap_server": "mail.doowoprnb.com",
        "imap_port": 993,
        "smtp_server": "mail.doowoprnb.com",
        "smtp_port": 465,
        "use_ssl": True,
        "password_env": "EMAIL_PASS_RUCKTALK",
    },
    "loovacast": {
        "email": "info@loovacast.com",
        "name": "LoovaCast",
        "imap_server": "mail.doowoprnb.com",
        "imap_port": 993,
        "smtp_server": "mail.doowoprnb.com",
        "smtp_port": 465,
        "use_ssl": True,
        "password_env": "EMAIL_PASS_LOOVACAST",
    },
    "lumabot": {
        "email": "lumabot@groundrushlabs.com",
        "name": "Luma Bot",
        "imap_server": "mail.doowoprnb.com",
        "imap_port": 993,
        "smtp_server": "mail.doowoprnb.com",
        "smtp_port": 465,
        "use_ssl": True,
        "password_env": "EMAIL_PASS_LUMABOT",
    },
    "support": {
        "email": "support@loovacast.com",
        "name": "LoovaCast Support",
        "imap_server": "mail.doowoprnb.com",
        "imap_port": 993,
        "smtp_server": "mail.doowoprnb.com",
        "smtp_port": 465,
        "use_ssl": True,
        "password_env": "EMAIL_PASS_SUPPORT_LOOVACAST",
    },
    "groundrush info": {
        "email": "info@groundrushlabs.com",
        "name": "Groundrush Info",
        "imap_server": "mail.doowoprnb.com",
        "imap_port": 993,
        "smtp_server": "mail.doowoprnb.com",
        "smtp_port": 465,
        "use_ssl": True,
        "password_env": "EMAIL_PASS_INFO_GROUNDRUSH",
    },
}


class EmailClient:
    """Multi-account email client using IMAP/SMTP."""

    def __init__(self):
        self._passwords = {}

    def _get_password(self, config: dict) -> str:
        """Get password for a specific account from environment."""
        env_key = config.get("password_env", "EMAIL_PASSWORD")
        if env_key not in self._passwords:
            self._passwords[env_key] = os.environ.get(env_key, "")
        return self._passwords[env_key]

    def _resolve_account(self, account: str) -> dict:
        """Resolve account name to config."""
        account_lower = account.lower().strip()

        # Direct lookup first
        if account_lower in EMAIL_ACCOUNTS:
            return EMAIL_ACCOUNTS[account_lower]

        # Normalize (remove special chars) for fuzzy matching
        key = account_lower.replace(" ", "").replace("-", "").replace("_", "")

        # Check normalized keys
        for alias, config in EMAIL_ACCOUNTS.items():
            alias_normalized = alias.replace("_", "")
            if key == alias_normalized:
                return config

        # Partial match on alias or email
        for alias, config in EMAIL_ACCOUNTS.items():
            if key in alias.replace("_", "") or key in config["email"].lower().replace("@", "").replace(".", ""):
                return config

        raise ValueError(f"Unknown account: {account}. Available: {', '.join(EMAIL_ACCOUNTS.keys())}")

    def _decode_header_value(self, value: str) -> str:
        """Decode email header value."""
        if not value:
            return ""
        decoded_parts = decode_header(value)
        result = []
        for part, charset in decoded_parts:
            if isinstance(part, bytes):
                result.append(part.decode(charset or "utf-8", errors="replace"))
            else:
                result.append(part)
        return "".join(result)

    def _parse_email(self, msg: email.message.Message, msg_id: str) -> dict:
        """Parse email message into dict."""
        subject = self._decode_header_value(msg.get("Subject", ""))
        from_addr = self._decode_header_value(msg.get("From", ""))
        to_addr = self._decode_header_value(msg.get("To", ""))
        date_str = msg.get("Date", "")

        # Parse date
        try:
            date = parsedate_to_datetime(date_str)
            date_formatted = date.strftime("%Y-%m-%d %H:%M")
        except:
            date_formatted = date_str

        # Extract body
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    try:
                        body = part.get_payload(decode=True).decode("utf-8", errors="replace")
                        break
                    except:
                        pass
        else:
            try:
                body = msg.get_payload(decode=True).decode("utf-8", errors="replace")
            except:
                body = str(msg.get_payload())

        return {
            "id": msg_id,
            "from": from_addr,
            "to": to_addr,
            "subject": subject,
            "date": date_formatted,
            "body": body[:5000],  # Limit body size
            "snippet": body[:200].replace("\n", " ").strip(),
        }

    def list_accounts(self) -> dict:
        """List all configured email accounts."""
        accounts = []
        for key, config in EMAIL_ACCOUNTS.items():
            accounts.append({
                "id": key,
                "email": config["email"],
                "name": config["name"],
            })
        return {"accounts": accounts, "count": len(accounts)}

    def get_inbox(self, account: str, limit: int = 10, folder: str = "INBOX") -> dict:
        """Get recent emails from an account's inbox."""
        try:
            config = self._resolve_account(account)
            password = self._get_password(config)

            if not password:
                return {"error": f"Email password not configured. Set {config.get('password_env')} in .env"}

            # Connect to IMAP
            if config["use_ssl"]:
                imap = imaplib.IMAP4_SSL(config["imap_server"], config["imap_port"])
            else:
                imap = imaplib.IMAP4(config["imap_server"], config["imap_port"])
                imap.starttls()

            imap.login(config["email"], password)
            imap.select(folder)

            # Search for recent messages
            status, messages = imap.search(None, "ALL")
            if status != "OK":
                imap.logout()
                return {"error": "Failed to search mailbox"}

            msg_ids = messages[0].split()
            # Get most recent
            msg_ids = msg_ids[-limit:] if len(msg_ids) > limit else msg_ids
            msg_ids.reverse()  # Most recent first

            emails = []
            for msg_id in msg_ids:
                status, data = imap.fetch(msg_id, "(RFC822)")
                if status == "OK":
                    msg = email.message_from_bytes(data[0][1])
                    parsed = self._parse_email(msg, msg_id.decode())
                    # Only include snippet for list view
                    emails.append({
                        "id": parsed["id"],
                        "from": parsed["from"],
                        "subject": parsed["subject"],
                        "date": parsed["date"],
                        "snippet": parsed["snippet"],
                    })

            imap.logout()

            return {
                "account": config["name"],
                "email": config["email"],
                "folder": folder,
                "messages": emails,
                "count": len(emails),
            }
        except Exception as e:
            logger.error(f"IMAP error for {account}: {e}")
            return {"error": str(e)}

    def read_email(self, account: str, message_id: str, folder: str = "INBOX") -> dict:
        """Read a full email by ID."""
        try:
            config = self._resolve_account(account)
            password = self._get_password(config)

            if not password:
                return {"error": f"Email password not configured. Set {config.get('password_env')} in .env"}

            if config["use_ssl"]:
                imap = imaplib.IMAP4_SSL(config["imap_server"], config["imap_port"])
            else:
                imap = imaplib.IMAP4(config["imap_server"], config["imap_port"])
                imap.starttls()

            imap.login(config["email"], password)
            imap.select(folder)

            status, data = imap.fetch(message_id.encode(), "(RFC822)")
            if status != "OK":
                imap.logout()
                return {"error": f"Message {message_id} not found"}

            msg = email.message_from_bytes(data[0][1])
            parsed = self._parse_email(msg, message_id)
            parsed["account"] = config["name"]

            imap.logout()
            return parsed
        except Exception as e:
            logger.error(f"Read email error for {account}: {e}")
            return {"error": str(e)}

    def send_email(self, account: str, to: str, subject: str, body: str, html: bool = False) -> dict:
        """Send an email from an account."""
        try:
            config = self._resolve_account(account)
            password = self._get_password(config)

            if not password:
                return {"error": f"Email password not configured. Set {config.get('password_env')} in .env"}

            # Create message
            if html:
                msg = MIMEMultipart("alternative")
                msg.attach(MIMEText(body, "html"))
            else:
                msg = MIMEText(body)

            msg["From"] = f"{config['name']} <{config['email']}>"
            msg["To"] = to
            msg["Subject"] = subject

            # Connect to SMTP
            if config["use_ssl"]:
                smtp = smtplib.SMTP_SSL(config["smtp_server"], config["smtp_port"])
            else:
                smtp = smtplib.SMTP(config["smtp_server"], config["smtp_port"])
                smtp.starttls()

            smtp.login(config["email"], password)
            smtp.send_message(msg)
            smtp.quit()

            logger.info(f"Email sent from {config['email']} to {to}: {subject}")
            return {
                "status": "sent",
                "from": config["email"],
                "to": to,
                "subject": subject,
            }
        except Exception as e:
            logger.error(f"Send email error for {account}: {e}")
            return {"error": str(e)}

    def search_emails(self, account: str, query: str, limit: int = 10, folder: str = "INBOX") -> dict:
        """Search emails in an account."""
        try:
            config = self._resolve_account(account)
            password = self._get_password(config)

            if not password:
                return {"error": f"Email password not configured. Set {config.get('password_env')} in .env"}

            if config["use_ssl"]:
                imap = imaplib.IMAP4_SSL(config["imap_server"], config["imap_port"])
            else:
                imap = imaplib.IMAP4(config["imap_server"], config["imap_port"])
                imap.starttls()

            imap.login(config["email"], password)
            imap.select(folder)

            # Build IMAP search criteria
            # Support simple keyword search in subject and from
            search_criteria = f'(OR SUBJECT "{query}" FROM "{query}")'

            status, messages = imap.search(None, search_criteria)
            if status != "OK":
                imap.logout()
                return {"error": "Search failed"}

            msg_ids = messages[0].split()
            msg_ids = msg_ids[-limit:] if len(msg_ids) > limit else msg_ids
            msg_ids.reverse()

            emails = []
            for msg_id in msg_ids:
                status, data = imap.fetch(msg_id, "(RFC822)")
                if status == "OK":
                    msg = email.message_from_bytes(data[0][1])
                    parsed = self._parse_email(msg, msg_id.decode())
                    emails.append({
                        "id": parsed["id"],
                        "from": parsed["from"],
                        "subject": parsed["subject"],
                        "date": parsed["date"],
                        "snippet": parsed["snippet"],
                    })

            imap.logout()

            return {
                "account": config["name"],
                "query": query,
                "messages": emails,
                "count": len(emails),
            }
        except Exception as e:
            logger.error(f"Search error for {account}: {e}")
            return {"error": str(e)}

    def get_unread_count(self, account: str) -> dict:
        """Get unread email count for an account."""
        try:
            config = self._resolve_account(account)
            password = self._get_password(config)

            if not password:
                return {"error": f"Email password not configured. Set {config.get('password_env')} in .env"}

            if config["use_ssl"]:
                imap = imaplib.IMAP4_SSL(config["imap_server"], config["imap_port"])
            else:
                imap = imaplib.IMAP4(config["imap_server"], config["imap_port"])
                imap.starttls()

            imap.login(config["email"], password)
            imap.select("INBOX")

            status, messages = imap.search(None, "UNSEEN")
            count = len(messages[0].split()) if status == "OK" and messages[0] else 0

            imap.logout()

            return {
                "account": config["name"],
                "email": config["email"],
                "unread": count,
            }
        except Exception as e:
            logger.error(f"Unread count error for {account}: {e}")
            return {"error": str(e)}

    def get_all_unread_counts(self) -> dict:
        """Get unread counts for all accounts."""
        results = []
        for key in EMAIL_ACCOUNTS:
            result = self.get_unread_count(key)
            if "error" not in result:
                results.append(result)
        return {"accounts": results}

    def trash_email(self, account: str, message_id: str, folder: str = "INBOX") -> dict:
        """Move an email to trash."""
        try:
            config = self._resolve_account(account)
            password = self._get_password(config)

            if not password:
                return {"error": f"Email password not configured. Set {config.get('password_env')} in .env"}

            if config["use_ssl"]:
                imap = imaplib.IMAP4_SSL(config["imap_server"], config["imap_port"])
            else:
                imap = imaplib.IMAP4(config["imap_server"], config["imap_port"])
                imap.starttls()

            imap.login(config["email"], password)
            imap.select(folder)

            # Try common trash folder names
            trash_folders = ["Trash", "INBOX.Trash", "Deleted", "INBOX.Deleted", "[Gmail]/Trash"]
            trash_folder = None

            # List folders to find trash
            status, folders = imap.list()
            if status == "OK":
                for f in folders:
                    f_decoded = f.decode() if isinstance(f, bytes) else f
                    for tf in trash_folders:
                        if tf.lower() in f_decoded.lower():
                            # Extract folder name from IMAP list response
                            parts = f_decoded.split('"')
                            if len(parts) >= 2:
                                trash_folder = parts[-2]
                            break
                    if trash_folder:
                        break

            if not trash_folder:
                trash_folder = "Trash"  # Default fallback

            # Copy to trash then delete from inbox
            status, _ = imap.copy(message_id.encode(), trash_folder)
            if status == "OK":
                imap.store(message_id.encode(), "+FLAGS", "\\Deleted")
                imap.expunge()
                imap.logout()
                return {
                    "status": "trashed",
                    "account": config["name"],
                    "message_id": message_id,
                    "moved_to": trash_folder,
                }
            else:
                # If copy fails, just mark as deleted
                imap.store(message_id.encode(), "+FLAGS", "\\Deleted")
                imap.expunge()
                imap.logout()
                return {
                    "status": "deleted",
                    "account": config["name"],
                    "message_id": message_id,
                }

        except Exception as e:
            logger.error(f"Trash email error for {account}: {e}")
            return {"error": str(e)}

    def mark_read(self, account: str, message_id: str, folder: str = "INBOX") -> dict:
        """Mark an email as read."""
        try:
            config = self._resolve_account(account)
            password = self._get_password(config)

            if not password:
                return {"error": f"Email password not configured. Set {config.get('password_env')} in .env"}

            if config["use_ssl"]:
                imap = imaplib.IMAP4_SSL(config["imap_server"], config["imap_port"])
            else:
                imap = imaplib.IMAP4(config["imap_server"], config["imap_port"])
                imap.starttls()

            imap.login(config["email"], password)
            imap.select(folder)

            imap.store(message_id.encode(), "+FLAGS", "\\Seen")
            imap.logout()

            return {
                "status": "marked_read",
                "account": config["name"],
                "message_id": message_id,
            }
        except Exception as e:
            logger.error(f"Mark read error for {account}: {e}")
            return {"error": str(e)}

    def mark_unread(self, account: str, message_id: str, folder: str = "INBOX") -> dict:
        """Mark an email as unread."""
        try:
            config = self._resolve_account(account)
            password = self._get_password(config)

            if not password:
                return {"error": f"Email password not configured. Set {config.get('password_env')} in .env"}

            if config["use_ssl"]:
                imap = imaplib.IMAP4_SSL(config["imap_server"], config["imap_port"])
            else:
                imap = imaplib.IMAP4(config["imap_server"], config["imap_port"])
                imap.starttls()

            imap.login(config["email"], password)
            imap.select(folder)

            imap.store(message_id.encode(), "-FLAGS", "\\Seen")
            imap.logout()

            return {
                "status": "marked_unread",
                "account": config["name"],
                "message_id": message_id,
            }
        except Exception as e:
            logger.error(f"Mark unread error for {account}: {e}")
            return {"error": str(e)}


# Singleton instance
email_client = EmailClient()
