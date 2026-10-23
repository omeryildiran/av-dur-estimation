"""Slack (or ntfy) updates while a participant runs (optional).

Setup (once): create a Slack app with an "Incoming Webhook" for the channel you want, and put
its URL in av_dur_experiment_v2/slack_webhook.txt (not in git) or in the environment variable
AVDUR_SLACK_WEBHOOK. Without either, nothing is sent. Instead of Slack, the file can hold an
ntfy topic URL (https://ntfy.sh/<a long secret name>; no account: subscribe to the same topic in
the ntfy app on your phone or computer). Test it:

    python -m avdur.notify "hello from the lab"

Messages carry only the participant ID and numbers. They are sent from a background thread with a
short timeout, at breaks and between blocks (never during a trial); a failure is ignored, so
Slack or the network can never stop or slow the experiment.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEBHOOK_FILE = ROOT / "slack_webhook.txt"
TIMEOUT_S = 5.0


def _url():
    url = os.environ.get("AVDUR_SLACK_WEBHOOK", "").strip()
    if not url and WEBHOOK_FILE.exists():
        url = WEBHOOK_FILE.read_text().strip()
    return url if url.startswith(("https://hooks.slack.com/", "https://ntfy.sh/")) else ""


def enabled():
    return bool(_url())


def _post(url, text):
    try:
        if "ntfy.sh/" in url:                               # ntfy: plain text body
            req = urllib.request.Request(url, data=text.encode("utf-8"), headers={"Title": "Experiment"})
        else:                                               # Slack incoming webhook: JSON
            req = urllib.request.Request(url, data=json.dumps({"text": text}).encode(),
                                         headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=TIMEOUT_S).read()
        return True
    except Exception:
        return False


def send(text, wait=False):
    """Post `text` to Slack in the background; wait=True blocks up to the timeout (use at exit)."""
    url = _url()
    if not url:
        return False
    text = f"[{socket.gethostname()}] {text}"
    if wait:
        return _post(url, text)
    threading.Thread(target=_post, args=(url, text), daemon=True).start()
    return True


if __name__ == "__main__":
    if not enabled():
        print(f"No webhook: put the URL in {WEBHOOK_FILE} or set AVDUR_SLACK_WEBHOOK.")
        sys.exit(1)
    ok = send(" ".join(sys.argv[1:]) or "test message from the experiment", wait=True)
    print("sent" if ok else "sending failed (check the URL and the internet connection)")
    sys.exit(0 if ok else 1)
