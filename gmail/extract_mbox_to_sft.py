#!/usr/bin/env python3
import argparse, mailbox, email, re, json, hashlib
from email.header import decode_header, make_header
from email.utils import parseaddr

SIG_SEP = re.compile(r"(?m)^\s*--\s*$")
FWD_ORIG = re.compile(r"(?mi)^(?:from:\s|sent:\s|subject:\s)|^(?:[-_]{5,}\s*original message\s*[-_]{5,})")
QUOTED = re.compile(r"(?m)^\s*>+")
WROTE_LINE = re.compile(r"(?mi)^on .*?wrote:\s*$")
HTML_TAG = re.compile(r"<[^>]+>")
WS = re.compile(r"[ \t]+")

def decode_hdr(h):
    try:
        return str(make_header(decode_header(h or "")))
    except Exception:
        return h or ""

def get_payload_text(msg):
    """Prefer text/plain; fallback to text/html → strip tags."""
    if msg.is_multipart():
        # first text/plain, else first text/html
        parts = list(msg.walk())
        for p in parts:
            if p.get_content_type() == "text/plain":
                return p.get_payload(decode=True) or b""
        for p in parts:
            if p.get_content_type() == "text/html":
                html = (p.get_payload(decode=True) or b"").decode(errors="ignore")
                return HTML_TAG.sub("", html).encode()
        return b""
    else:
        payload = msg.get_payload(decode=True)
        if payload is None:
            # may be str if not encoded
            p = msg.get_payload()
            if isinstance(p, str):
                return p.encode()
            return b""
        if msg.get_content_type() == "text/html":
            return HTML_TAG.sub("", payload.decode(errors="ignore")).encode()
        return payload

def clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # cut signature
    s = SIG_SEP.split(s, 1)[0]
    # cut after typical quoted markers
    s = re.split(WROTE_LINE, s, maxsplit=1)[0]
    s = FWD_ORIG.split(s, 1)[0]
    # drop quoted lines
    s = "\n".join(line for line in s.splitlines() if not QUOTED.match(line))
    # collapse whitespace
    s = WS.sub(" ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s

def same_person(a, b):
    return parseaddr(a)[1].lower() == parseaddr(b)[1].lower()

def extract_threads(mbox_path, my_email, min_chars=40, max_chars=4000):
    box = mailbox.mbox(mbox_path)
    # index by Message-ID for quick lookup, and gather threading links
    by_id = {}
    for m in box:
        mid = (m.get("Message-ID") or "").strip()
        if mid:
            by_id[mid] = m

    samples = []
    seen = set()

    for m in box:
        # We want messages WE sent that reply to someone: In-Reply-To present and From == my_email
        if not same_person(m.get("From", ""), my_email):
            continue
        in_reply_to = (m.get("In-Reply-To") or "").strip()
        if not in_reply_to or in_reply_to not in by_id:
            continue

        # our reply (output)
        out_bytes = get_payload_text(m)
        out_txt = clean_text(out_bytes.decode(errors="ignore"))

        # their email we replied to (input)
        prev = by_id[in_reply_to]
        inp_bytes = get_payload_text(prev)
        inp_txt = clean_text(inp_bytes.decode(errors="ignore"))

        # skip bad pairs
        if not inp_txt or not out_txt:
            continue
        if not (min_chars <= len(out_txt) <= max_chars):
            continue

        # dedupe by content hash
        h = hashlib.sha1((inp_txt + "\n---\n" + out_txt).encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)

        samples.append({
            "instruction": "Reply to the following email in my tone and style.",
            "input": inp_txt,
            "output": out_txt,
        })

    return samples

def main():
    ap = argparse.ArgumentParser(description="Convert Gmail MBOX → instruction/input/output JSONL")
    ap.add_argument("mbox", help="Path to Takeout .mbox")
    ap.add_argument("--my_email", required=True, help="Your email address (identifies your replies)")
    ap.add_argument("-o", "--out", default="gmail_sft_instruct.jsonl", help="Output JSONL file")
    ap.add_argument("--min_chars", type=int, default=40)
    ap.add_argument("--max_chars", type=int, default=4000)
    args = ap.parse_args()

    samples = extract_threads(args.mbox, args.my_email, args.min_chars, args.max_chars)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {len(samples)} samples → {args.out}")
    # show a tiny preview
    for i, s in enumerate(samples[:3]):
        print(f"\n--- sample {i+1} ---")
        print("INSTRUCTION:", s["instruction"])
        print("INPUT      :", (s['input'][:200] + "…") if len(s['input']) > 200 else s['input'])
        print("OUTPUT     :", (s['output'][:200] + "…") if len(s['output']) > 200 else s['output'])

if __name__ == "__main__":
    main()
