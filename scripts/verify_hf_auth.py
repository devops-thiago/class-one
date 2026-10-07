#!/usr/bin/env python3
from huggingface_hub import HfApi

with open(".env", encoding="utf-8") as f:
    token = [line.split("=", 1)[1].strip().strip("'\"") for line in f if line.startswith("HF_TOKEN=")][0]

api = HfApi(token=token)
try:
    user = api.whoami()
    print("Authenticated with HF as:", user.get("name"))
except Exception as e:
    print("HF auth error:", e)
