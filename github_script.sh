#!/bin/bash
# Usage: ./setup_gpu_agent.sh <github_pat> <branch_name>

GITHUB_PAT=$1
BRANCH_NAME=$2
REPO="JayZenith/vllm"
KEY_FILE="$HOME/.ssh/id_ed25519"

# 1. Generate SSH key
ssh-keygen -t ed25519 -f "$KEY_FILE" -N "" -C "agent@$HOSTNAME"

# 2. Add deploy key to GitHub fork
PUB_KEY=$(cat "${KEY_FILE}.pub")
curl -s -X POST -H "Authorization: token $GITHUB_PAT" \
     -H "Accept: application/vnd.github+json" \
     https://api.github.com/repos/$REPO/keys \
     -d "{\"title\":\"agent@$HOSTNAME\",\"key\":\"$PUB_KEY\",\"read_only\":false}"

# 3. Start SSH agent and add key
eval "$(ssh-agent -s)"
ssh-add "$KEY_FILE"

# 4. Add GitHub to known hosts to skip prompt
ssh-keyscan github.com >> ~/.ssh/known_hosts

# 5. Clone fork and create branch
git clone git@github.com:$REPO.git
cd $(basename "$REPO")
git checkout -b "$BRANCH_NAME"

echo "✅ GPU agent setup complete. Working on branch '$BRANCH_NAME'."

