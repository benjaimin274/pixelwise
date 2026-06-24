#!/bin/bash
# deploy/auto-deploy.sh
# Authored on dev, executed on prod by the systemd timer scheduler loops.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd /opt/pixelwise

BRANCH="${DEPLOY_BRANCH:-feature/vm-migration-dual-ood}"

# Fetch latest history tracking points from git for the configured branch
git fetch origin "$BRANCH"

# Extract comparative verification commit fingerprints
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/"$BRANCH")

# If no new changes are found upstream, exit cleanly and cheaply
if [ "$LOCAL" = "$REMOTE" ]; then
    exit 0
fi

echo "New change found on upstream tracking matrix: $REMOTE"

# Pull down the tracking modifications
git checkout "$BRANCH"
git pull origin "$BRANCH"

# Activate environment and synchronize packages
source .venv/bin/activate
pip install -r requirements.txt > /dev/null

# Execute the validation gate test suite (using python -m to resolve paths)
if ! python -m pytest tests/; then
    echo "CRITICAL: Automated tests failed. Aborting deployment cycle to protect production."
    exit 1
fi

echo "Syncing frontend static assets to Nginx web root..."
sudo cp -r "$SCRIPT_DIR/frontend/"* /var/www/pixelwise/
KEY=$(grep ^SECRET_API_KEY "$SCRIPT_DIR/.env" | cut -d'=' -f2)
sudo sed -i "s/REPLACE_ME/$KEY/" /var/www/pixelwise/app.js

# If the code passes validation, trigger a safe application service restart
echo "Validation passed. Restarting live service modules..."
sudo systemctl restart pixelwise
echo "Successfully deployed tracking commit change: $REMOTE"
