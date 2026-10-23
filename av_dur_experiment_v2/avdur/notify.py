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
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEBHOOK_FILE = ROOT / "slack_webhook.txt"
TIMEOUT_S = 5.0
_THREADS = []


def _background(fn, *args):
    t = threading.Thread(target=fn, args=args, daemon=True)
    t.start()
    _THREADS.append(t)


def flush(timeout=20.0):
    """Wait (up to `timeout` s in total) for messages still being sent; call before the program exits."""
    import time
    end = time.monotonic() + timeout
    for t in list(_THREADS):
        t.join(max(0.0, end - time.monotonic()))


def _url():
    if os.environ.get("AVDUR_NO_NOTIFY"):                 # set by the tests
        return ""
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


BOT_FILE = ROOT / "slack_bot.txt"          # line 1: bot token (xoxb-...), line 2: channel ID (C...)


def _bot():
    if os.environ.get("AVDUR_NO_NOTIFY") or not BOT_FILE.exists():
        return None
    lines = [x.strip() for x in BOT_FILE.read_text().splitlines() if x.strip()]
    return (lines[0], lines[1]) if len(lines) >= 2 and lines[0].startswith("xoxb-") else None


def _api(method, token, data=None, json_body=None):
    url = f"https://slack.com/api/{method}"
    if json_body is not None:
        req = urllib.request.Request(url, data=json.dumps(json_body).encode(),
                                     headers={"Authorization": f"Bearer {token}",
                                              "Content-Type": "application/json; charset=utf-8"})
    else:
        req = urllib.request.Request(url, data=urllib.parse.urlencode(data).encode(),
                                     headers={"Authorization": f"Bearer {token}"})
    return json.loads(urllib.request.urlopen(req, timeout=TIMEOUT_S * 3).read())


def _upload(path, comment):
    """Slack (bot token, files:write) or ntfy attachment; False if neither is set up or it fails."""
    try:
        path = Path(path)
        data = path.read_bytes()
        bot = _bot()
        if bot:
            token, channel = bot
            r = _api("files.getUploadURLExternal", token, data={"filename": path.name, "length": len(data)})
            if not r.get("ok"):
                return False
            urllib.request.urlopen(urllib.request.Request(r["upload_url"], data=data), timeout=TIMEOUT_S * 3).read()
            done = _api("files.completeUploadExternal", token, json_body={
                "files": [{"id": r["file_id"], "title": path.name}], "channel_id": channel,
                "initial_comment": comment or path.name})
            return bool(done.get("ok"))
        url = _url()
        if url.startswith("https://ntfy.sh/"):
            req = urllib.request.Request(url, data=data, method="PUT",
                                         headers={"Filename": path.name,          # headers: ASCII only
                                                  "Title": comment.encode("ascii", "ignore").decode().strip(" -")[:200]
                                                  or path.name})
            urllib.request.urlopen(req, timeout=TIMEOUT_S * 3).read()
            return True
    except Exception:
        pass
    return False


def block_done(pid, text, plots):
    """One message per finished block: its summary (without file paths), then its plots."""
    text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("plot:"))
    send_with_plots(f"✅ {pid}  " + text.replace("──", "").strip(), plots)


def send_with_plots(text, plots):
    """The text (webhook) and then each existing plot (bot upload or ntfy), all in the background."""
    send(text)
    plots = [p for p in plots if p and Path(p).exists()]
    if plots and (_bot() or _url().startswith("https://ntfy.sh/")):
        first_line = text.splitlines()[0] if text else ""
        _background(lambda: [_upload(p, first_line) for p in plots])


def send(text, wait=False):
    """Post `text` to Slack in the background; wait=True blocks up to the timeout (use at exit)."""
    url = _url()
    if not url:
        return False
    text = f"[{socket.gethostname()}] {text}"
    if wait:
        return _post(url, text)
    _background(_post, url, text)
    return True


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--image":            # python -m avdur.notify --image plot.png
        ok = _upload(sys.argv[2], "test image from the experiment")
        print("image sent" if ok else f"image upload failed: needs {BOT_FILE.name} (Slack bot) or an ntfy URL")
        sys.exit(0 if ok else 1)
    if not enabled():
        print(f"No webhook: put the URL in {WEBHOOK_FILE} or set AVDUR_SLACK_WEBHOOK.")
        sys.exit(1)
    ok = send(" ".join(sys.argv[1:]) or "test message from the experiment", wait=True)
    print("sent" if ok else "sending failed (check the URL and the internet connection)")
    sys.exit(0 if ok else 1)
