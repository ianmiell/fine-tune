# fine-tune

A script to follow along with the https://zwischenzugs.com blog to fine-tune a huggingface image against your own data (in the example here, sent gmails).

It uses [vast.ai](https://cloud.vast.ai/?ref_id=341917) to provision GPUs (note the link is a referral link I benefit from if you sign up).

## Setup / Requirements


### Get gmail mbox

Use Google Takeout to download your sent email to a `.mbox` file. You probably only need your 'Sent' emails, as they contain your responses.

Then run the gmail extraction script.

`python ./gmail/extract_mbox_to_sft.py --my_email youremailadress@example.com -o ./gmail/gmail_sft_sharegpt.jsonl /path/to/mbox/file/Sent.mbox`

and then run:

`xz gmail/gmail_sft_instruct.jsonl`

to compress it.

### Huggingface account and login tokens required

Set up an account on [huggingface](https://huggingface.co/) and get a write token from [here](https://huggingface.co/settings/tokens).

File: `huggingface_token` - (in the root dir of this repo) should contain a write token. Git will ignore it, so it won't be leaked.

### Venv setup

In the repo root, run:

```
python -m venv .venv || python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Set up vastai

See [here](https://docs.vast.ai/cli/get-started?ref_id=341917) for help setting up.

Run:

`vastai set api-key xxxxxxxxxxxxxxxxxxxxxxxxxx`

## Running

`./run_fine_tune.sh --hf-user YOUR_HUGGINGFACE_USERNAME`

Run:

`./run_fine_tune.sh -h`

for options.

