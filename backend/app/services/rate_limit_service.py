import logging
import time
from fastapi import HTTPException, status
from typing import Optional
from app.config.settings import settings

logger = logging.getLogger(__name__)

redis_client = None
if settings.UPSTASH_REDIS_REST_URL and settings.UPSTASH_REDIS_REST_TOKEN:
    from upstash_redis import Redis
    try:
        redis_client = Redis(url=settings.UPSTASH_REDIS_REST_URL, token=settings.UPSTASH_REDIS_REST_TOKEN)
    except Exception as e:
        logger.error(f"Failed to initialize Upstash Redis: {e}")

class RateLimitService:
    def __init__(self):
        self.redis = redis_client

    def check_limit(self, key: str, max_requests: int, window_seconds: int, error_message: str):
        if not self.redis:
            logger.warning("Redis not configured. Skipping rate limit.")
            return

        current_time = int(time.time())
        window_key = f"rl:{key}:{current_time // window_seconds}"
        
        try:
            # Fixed window rate limiting
            count = self.redis.incr(window_key)
            if count == 1:
                self.redis.expire(window_key, window_seconds)
                
            if count > max_requests:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=error_message,
                    headers={"Retry-After": str(window_seconds)}
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Rate limiting failed for key {key}: {e}")

    def enforce_otp_send_limits(self, email: str, ip: str):
        # 1 request per 60 seconds per email
        self.check_limit(f"otp:60s:{email}", 1, 60, "Please wait 60 seconds before requesting another code.")
        
        # 5 OTP sends per hour per email
        self.check_limit(f"otp:1h:{email}", 5, 3600, "Too many OTP requests. Please try again in an hour.")
        
        # 10 OTP requests per 10 minutes per IP
        if ip:
            self.check_limit(f"otp:10m:ip:{ip}", 10, 600, "Too many requests from this IP. Please try again later.")
            
        # 10 OTP requests per 24 hours per email
        self.check_limit(f"otp:24h:{email}", 10, 86400, "Daily limit reached for this email. Please try again tomorrow.")

    def enforce_password_reset_limits(self, email: str, ip: str):
        # 1 request per 60 seconds per email
        self.check_limit(f"pwdreset:60s:{email}", 1, 60, "Please wait 60 seconds before requesting another reset link.")
        
        # 5 password-reset OTP requests per hour per email
        self.check_limit(f"pwdreset:1h:{email}", 5, 3600, "Too many reset requests. Please try again in an hour.")
        
        # 10 reset requests per 10 minutes per IP
        if ip:
            self.check_limit(f"pwdreset:10m:ip:{ip}", 10, 600, "Too many reset requests from this IP. Please try again later.")

    def enforce_verify_limits(self, email: str, ip: str):
        # 20 verify requests per 10 minutes per email
        self.check_limit(f"verify:10m:{email}", 20, 600, "Too many verification attempts. Please try again later.")
        
        if ip:
            self.check_limit(f"verify:10m:ip:{ip}", 30, 600, "Too many requests from this IP. Please try again later.")

rate_limit_service = RateLimitService()
