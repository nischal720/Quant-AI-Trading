# utils.py
import re
import time
from collections import defaultdict
from functools import wraps
from typing import Callable, Dict, List, Optional, Tuple
import logging
import aiohttp
from tenacity import retry, wait_exponential, stop_after_attempt
from flask import request
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

flask_rate_limits = defaultdict(lambda: {"count": 0, "reset_time": 0})
telegram_rate_limits = defaultdict(lambda: {"count": 0, "reset_time": 0})

def rate_limit(limit_type: str):
    def decorator(f: Callable):
        @wraps(f)
        async def wrapped(*args, **kwargs):
            from config import CONFIG
            client_ip = request.remote_addr if limit_type == "flask" else "telegram"
            limits = CONFIG[f"{limit_type}_rate_limit"]
            now = time.time()
            rate_info = flask_rate_limits[client_ip] if limit_type == "flask" else telegram_rate_limits[client_ip]
            
            if now > rate_info["reset_time"]:
                rate_info["count"] = 0
                rate_info["reset_time"] = now + limits["window"]
            
            if rate_info["count"] >= limits["requests"]:
                logger.warning(f"Rate limit exceeded for {client_ip} ({limit_type})")
                if limit_type == "flask":
                    return json.dumps({"error": "Rate limit exceeded"}), 429
                return False
            
            rate_info["count"] += 1
            return await f(*args, **kwargs)
        return wrapped
    return decorator

def sanitize_string(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\w\s.,:;@#$%^&*()_+-=]', '', text)
    return text[:500]

@retry(wait=wait_exponential(multiplier=2, min=2, max=30), stop=stop_after_attempt(15))
async def fetch_json(url: str) -> Dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                return await response.json()
            raise Exception(f"HTTP {response.status}")