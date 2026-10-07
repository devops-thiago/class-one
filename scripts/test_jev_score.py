#!/usr/bin/env python3
"""Check score question format on Jev API."""

import json
import socket
import urllib.error
import urllib.request

old_getaddrinfo = socket.getaddrinfo


def new_getaddrinfo(host, port, *args, **kwargs):
    if host == "api.typesafe.ai":
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("104.18.26.46", port))]
    return old_getaddrinfo(host, port, *args, **kwargs)


socket.getaddrinfo = new_getaddrinfo

with open(".env", encoding="utf-8") as f:
    token = [line.split("=", 1)[1].strip().strip("'\"") for line in f if line.startswith("JEV_TOKEN=")][0]


def test_payload(name, q_payload):
    req_data = {
        "model": "jev-latest",
        "state": "The application crashed with an uncaught NullPointerException in the payment flow.",
        "questions": {"severity": q_payload},
    }
    req = urllib.request.Request(
        "https://api.typesafe.ai/v1/systemone",
        data=json.dumps(req_data).encode("utf-8"),
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print(f"[✓] {name} SUCCESS:")
            print(json.dumps(data["answers"]["severity"], indent=2))
            return True
    except urllib.error.HTTPError as e:
        print(f"[!] {name} FAILED HTTP {e.code}: {e.read().decode('utf-8')[:200]}")
        return False


# Try format 1: "criteria": ["level1", "level2", "level3"]
test_payload(
    "criteria list",
    {
        "type": "score",
        "instructions": "Rate severity:",
        "criteria": ["minor inconvenience", "moderate disruption", "critical payment failure"],
    },
)

# Try format 2: "levels": ["level1", "level2", "level3"]
test_payload(
    "levels list",
    {
        "type": "score",
        "instructions": "Rate severity:",
        "levels": ["minor inconvenience", "moderate disruption", "critical payment failure"],
    },
)

# Try format 3: "levels": {"1": "...", "2": "...", "3": "..."}
test_payload(
    "levels dict",
    {
        "type": "score",
        "instructions": "Rate severity:",
        "levels": {"1": "minor inconvenience", "2": "moderate disruption", "3": "critical payment failure"},
    },
)
