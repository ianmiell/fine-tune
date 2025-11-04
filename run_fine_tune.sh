#!/usr/bin/env bash

set -eo pipefail

# Default values (overridable through CLI flags)
GMAIL_JSONL_FILENAME=gmail/gmail_sft_instruct.jsonl
DISK_GB=64                        # container disk size
GPU_NAME="RTX 5090"               # 5090 is roughly 3 times the speed of training of 3090
RESULT_LIMIT=5
MAX_STEPS=2000
FINE_TUNE_BASE_MODEL=Qwen/Qwen2.5-3B-Instruct

usage() {
  cat << EOF
Usage: build_model.sh [options]

Options:
  --hf-user NAME         Huggingface username (required)
  --gmail-jsonl PATH     Path to the JSONL dataset (without trailing .xz)
                        (default: ${GMAIL_JSONL_FILENAME}
  --disk-gb SIZE         Disk size in GB for the instance (default: ${DISK_DB})
  --gpu-name NAME        GPU name to request from Vast (default: ${GPU_NAME})
  --base-model NAME      Base model, (default: ${FINE_TUNE_BASE_MODEL})
  --max-steps SIZE       Number of max steps. Setting to low value (eg 1) runs
                         much more useful when testing, as it does a minimal
                         amount of training. (default: ${MAX_STEPS})
  --help                 Show this help message and exit
EOF
}

# Default values (overridable through CLI flags)
GMAIL_JSONL_FILENAME=gmail/gmail_sft_instruct.jsonl
DISK_GB=64                        # container disk size
GPU_NAME="RTX 5090"               # 5090 is roughly 3 times the speed of training of 3090
RESULT_LIMIT=5
MAX_STEPS=2000
FINE_TUNE_BASE_MODEL=Qwen/Qwen2.5-3B-Instruct

# Parse options
while [[ "$1" != '' ]]; do
  case "$1" in
    --hf-user)
      HF_USERNAME="$2"; shift ;;
    --max-steps)
      MAX_STEPS="$2"; shift ;;
    --gmail-jsonl)
      GMAIL_JSONL_FILENAME="$2"; shift ;;
    --disk-gb)
      DISK_GB="$2"; shift ;;
    --gpu-name)
      GPU_NAME="$2"; shift ;;
    --base-model)
      FINE_TUNE_BASE_MODEL="$2"; shift ;;
    -h|--help)
      usage; exit 0 ;;
    --)
      shift;;
    *)
      echo "Error: Unknown option $1"
      usage; exit 1 ;;
  esac
  shift
done

set -u

# Check all is set up correctly
[ -n $HF_USERNAME ] || ( usage; exit 1 )
command vastai || ( echo "Please install vast.ai cli tool: https://docs.vast.ai/cli/get-started"; exit 1 )
[ -n $VIRTUAL_ENV ] || ( echo "Please set up venv (see README.md)"; exit 1 )
[ pip check ] || ( echo "Please install pip dependencies as per requirements.txt (see README.md)"; exit 1 )
[ -f gmail/gmail_sft_sharegpt.jsonl.xz ] || ( echo "Please create and/or compress gmail_sft_sharegpt.jsonl file (see README.md)"; exit 1 )

# Example debug output
echo "MAX_STEPS=$MAX_STEPS"
echo "GMAIL_JSONL_FILENAME=$GMAIL_JSONL_FILENAME"
echo "DISK_GB=$DISK_GB"
echo "GPU_NAME=$GPU_NAME"
echo "HF_USERNAME=$HF_USERNAME"

# Build the Vast filter (RTX + your constraints)
QUERY=$(
  cat <<Q
gpu_arch = "nvidia" rentable = true  verified = true  cpu_ram > 32  cpu_ghz > 1.0  inet_down >= 1000  inet_up >= 1000  disk_space >= ${DISK_GB}  min_bid > 0  gpu_name = "$GPU_NAME"
Q
)
unset GPU_NAME

echo ">> Searching Vast.ai offers with query: ${QUERY}"

# We ask Vast to return a raw table (TSV) sorted by min_bid (spot price).
# NOTE: --raw outputs a tab-separated table with a header line.

SEARCH_OUT="$(vastai search offers -d -o min_bid --raw --limit "${RESULT_LIMIT}" "$QUERY")"
unset QUERY

# Sanity
echo "Costs found"
echo "$SEARCH_OUT" | jq '.[0].min_bid'
if [[ -z "$SEARCH_OUT" ]]; then
  echo "!! No results returned by Vast CLI." >&2
  exit 1
fi

VAST_ID="$(echo "$SEARCH_OUT" | jq '.[0].id')"
unset SEARCH_OUT
CREATION_STRING="$(vastai create instance "$VAST_ID" --image pytorch/pytorch --disk "${DISK_GB}")"
unset VAST_ID DISK_GB
IID=$(echo "$CREATION_STRING" | sed -E 's/.*new_contract...([0-9]+).*/\1/')
unset CREATION_STRING

# Wait for it to attach?
until vastai show instance "$IID" --raw | grep 'actual_status.*running'; do
  echo "waiting for instance to be runnable..."
  echo "vastai show instance $IID --raw"
  STATUS_MSG=$(vastai show instance "$IID" --raw | grep 'status_msg.*Error' || true)
  echo "Status: $STATUS_MSG"
  if echo "$STATUS_MSG" | grep 'status_msg.*Error'
  then
    echo ERROR - destroy instance!?
    vastai destroy instance "$IID"
    exit 1
  fi
  sleep 20
done
unset STATUS_MSG

vastai attach ssh "$IID" "$(cat ~/.ssh/id_ed25519.pub)"
SCP_URL="$(vastai scp-url "$IID")"
SSH_URL="$(vastai ssh-url "$IID")"

CMD_SCP_HUGGINGFACE_TOKEN=$(echo "$SCP_URL" | sed -E 's#scp://([^@]+)@([^:]+):([0-9]+)#scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P \3 huggingface_token \1@\2:/workspace#')
CMD_SCP_RUN_SCRIPT=$(echo "$SCP_URL" | sed -E 's#scp://([^@]+)@([^:]+):([0-9]+)#scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P \3 build_and_push_model.sh \1@\2:/workspace#')
# shellcheck disable=SC2016
CMD_SCP_GMAIL_JSONL=$(echo "$SCP_URL" | sed -E 's#scp://([^@]+)@([^:]+):([0-9]+)#scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P \3 ${GMAIL_JSONL_FILENAME}.xz \1@\2:/workspace#')
echo "Uploading: $CMD_SCP_HUGGINGFACE_TOKEN"
until eval "$CMD_SCP_HUGGINGFACE_TOKEN"; do sleep 5; done
echo "Uploading: $CMD_SCP_RUN_SCRIPT"
until eval "$CMD_SCP_RUN_SCRIPT"; do sleep 5; done

echo "Uploading: $CMD_SCP_GMAIL_JSONL"
until eval "$CMD_SCP_GMAIL_JSONL"; do sleep 5; done
unset CMD_SCP_HUGGINGFACE_TOKEN
unset CMD_SCP_RUN_SCRIPT
unset CMD_SCP_GMAIL_JSONL

CMD_SSH_RUN_SCRIPT=$(echo "$SSH_URL" | sed -E "s#ssh://([^@]+)@([^:]+):([0-9]+)#ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p \3 \1@\2 HF_USERNAME=${HF_USERNAME} FINE_TUNE_BASE_MODEL=${FINE_TUNE_BASE_MODEL} LLAMAFACTORY_TRAIN_MAX_STEPS=${MAX_STEPS} /workspace/build_and_push_model.sh#")
echo "Running: $CMD_SSH_RUN_SCRIPT"
until eval "$CMD_SSH_RUN_SCRIPT"; do sleep 5; done
unset CMD_SSH_RUN_SCRIPT


echo "$SSH_URL"

echo "Destroy instance (y/n)?"
read -r ANSWER
if [[ $ANSWER == "y" ]]
then
  vastai destroy instance "${IID}"
fi
unset SCP_URL SSH_URL
