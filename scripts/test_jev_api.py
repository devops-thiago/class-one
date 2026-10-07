#!/usr/bin/env python3
"""Test script for TypeSafe AI Jev API connectivity via routed IP."""

import json
import os
import socket
import sys
import urllib.error
import urllib.request

# Route api.typesafe.ai to reachable Cloudflare edge IP 104.18.26.46 (bypassing ISP hop timeout on 104.18.24.46)
old_getaddrinfo = socket.getaddrinfo


def new_getaddrinfo(host, port, *args, **kwargs):
    if host == "api.typesafe.ai":
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("104.18.26.46", port))]
    return old_getaddrinfo(host, port, *args, **kwargs)


socket.getaddrinfo = new_getaddrinfo


def get_token():
    token = os.environ.get("JEV_TOKEN") or os.environ.get("TYPESAFE_API_KEY")
    if token:
        return token
    if os.path.exists(".env"):
        with open(".env", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("JEV_TOKEN=") or line.startswith("TYPESAFE_API_KEY="):
                    val = line.split("=", 1)[1].strip()
                    val = val.strip("'").strip('"')
                    return val
    return None


def main():
    token = get_token()
    if not token:
        print("[!] No JEV_TOKEN found.")
        sys.exit(1)

    print("[*] JEV_TOKEN found. Connecting to https://api.typesafe.ai/v1/systemone via 104.18.26.46 ...")

    req_data = {
        "model": "jev-latest",
        "state": "Customer requested a full refund for an unreceived package.",
        "questions": {
            "is_urgent": {
                "type": "noul",
                "instructions": "Is this issue time-critical or urgent?",
            },
            "category": {
                "type": "choice",
                "instructions": "Classify the ticket topic.",
                "criteria": {
                    "billing": "Invoices, charges, refunds",
                    "shipping": "Tracking, delayed transit, delivery issues",
                },
            },
        },
    }

    req = urllib.request.Request(
        "https://api.typesafe.ai/v1/systemone",
        data=json.dumps(req_data).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
            "User-Agent": "classone-benchmark/0.1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            print(f"[✓] HTTP {response.status} OK")
            body = json.loads(response.read().decode("utf-8"))
            print("[✓] Response received from Jev API:")
            print(json.dumps(body, indent=2))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        print(f"[!] HTTP Error {e.code}: {err_body}")
    except Exception as e:
        print(f"[!] Request error: {e}")


if __name__ == "__main__":
    main()
