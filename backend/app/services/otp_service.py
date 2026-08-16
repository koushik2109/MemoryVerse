import secrets
import hmac
import hashlib
import time
import json
import logging
from typing import Optional, Dict, Any
from app.config.settings import settings
from app.services.rate_limit_service import redis_client

logger = logging.getLogger(__name__)

# Constants for OTP Purposes
PURPOSE_EMAIL_VERIFICATION = "EMAIL_VERIFICATION"
PURPOSE_PASSWORD_RESET = "PASSWORD_RESET"

class OTPService:
    def __init__(self):
        self.redis = redis_client
        self.secret = settings.OTP_HASH_SECRET.encode('utf-8')
        
    def _hash_otp(self, otp: str) -> str:
        """Hash OTP using HMAC-SHA256 for brute force / leakage protection."""
        return hmac.new(self.secret, otp.encode('utf-8'), hashlib.sha256).hexdigest()

    def generate_otp(self) -> str:
        """Generate a cryptographically secure 6-digit OTP."""
        # Generates a random integer between 0 and 999999, formats to 6 digits with leading zeros
        return f"{secrets.randbelow(1000000):06d}"
        
    def store_otp(self, email: str, purpose: str, otp: str) -> bool:
        """Store hashed OTP with metadata in Redis."""
        if not self.redis:
            logger.error("Redis is not configured. Cannot store OTP.")
            return False
            
        otp_hash = self._hash_otp(otp)
        key = f"otp:{purpose}:{email}"
        
        state = {
            "hash": otp_hash,
            "attempts": 0,
            "expires_at": int(time.time()) + (settings.OTP_EXPIRE_MINUTES * 60)
        }
        
        try:
            # Overwrites any existing OTP for this email and purpose immediately (invalidates old)
            self.redis.set(key, json.dumps(state), ex=settings.OTP_EXPIRE_MINUTES * 60)
            return True
        except Exception as e:
            logger.error(f"Failed to store OTP in Redis: {e}")
            return False

    def verify_otp(self, email: str, purpose: str, submitted_otp: str) -> tuple[bool, str]:
        """
        Verify the OTP against stored state.
        Returns (is_valid, message)
        """
        if not self.redis:
            return False, "OTP service unavailable."
            
        key = f"otp:{purpose}:{email}"
        
        try:
            state_str = self.redis.get(key)
            if not state_str:
                return False, "OTP expired or does not exist."
                
            state = json.loads(state_str)
            
            # Check expiration
            if int(time.time()) > state.get("expires_at", 0):
                self.redis.delete(key)
                return False, "OTP expired."
                
            # Check attempts
            attempts = state.get("attempts", 0)
            if attempts >= settings.OTP_MAX_ATTEMPTS:
                self.redis.delete(key)
                return False, "Maximum attempts reached. Please request a new code."
                
            # Verify hash securely
            submitted_hash = self._hash_otp(submitted_otp)
            if not hmac.compare_digest(submitted_hash, state["hash"]):
                # Increment failed attempts
                state["attempts"] = attempts + 1
                if state["attempts"] >= settings.OTP_MAX_ATTEMPTS:
                    self.redis.delete(key)
                    return False, "Incorrect code. Maximum attempts reached. Please request a new code."
                else:
                    # Update state in Redis but preserve TTL
                    ttl = self.redis.ttl(key)
                    if ttl > 0:
                        self.redis.set(key, json.dumps(state), ex=ttl)
                    return False, f"Incorrect code. {settings.OTP_MAX_ATTEMPTS - state['attempts']} attempts remaining."
                    
            # Success: invalidate OTP immediately to prevent reuse
            self.redis.delete(key)
            return True, "Success"
            
        except Exception as e:
            logger.error(f"Error verifying OTP: {e}")
            return False, "Internal error verifying OTP."

otp_service = OTPService()
