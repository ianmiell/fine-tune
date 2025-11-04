# ai-experimentation

## Setup

### Get gmail mbox

Use Google Takeout to download your sent email to a `.mbox` file. You probably only need your 'Sent' emails, as they contain your responses.

Then run the gmail extraction script.

`python ./gmail/extract_mbox_to_sft.py --my_email youremailadress@example.com -o ./gmail/gmail_sft_sharegpt.jsonl /path/to/mbox/file/Sent.mbox`

and then run:

`xz gmail/gmail_sft_instruct.jsonl`

to compress it.

### Login tokens required

`huggingface_token` - in root dir, should contain a write token. Git will ignore it.

### Venv setup

```
python -m venv .venv || python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

`./run_fine_tune.sh`

Run:

`./run_fine_tune.sh -h`

for options.
