#!/bin/bash
# deploy/auto-deploy.sh
# Authored on dev, executed on prod by the systemd timer scheduler loops.
set -euo pipefail

cd /opt/pixelwise

# Fetch latest history tracking points from git
git fetch origin

# Extract comparative verification commit fingerprints
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/feature/vm-migration-dual-ood)

# If no new changes are found upstream, exit cleanly and cheaply
if [ "$LOCAL" = "$REMOTE" ]; then
    exit 0
fi

echo "New change found on upstream tracking matrix: $REMOTE"

# Pull down the tracking modifications
git pull origin feature/vm-migration-dual-ood

# Activate environment and synchronize packages
source .venv/bin/activate
pip install -r requirements.txt > /dev/null

# Execute the validation gate test suite (using python -m to resolve paths)
if ! python -m pytest tests/; then
    echo "CRITICAL: Automated tests failed. Aborting deployment cycle to protect production."
    exit 1
fi

# If the code passes validation, trigger a safe application service restart
echo "Validation passed. Restarting live service modules..."
sudo systemctl restart pixelwise
echo "Successfully deployed tracking commit change: $REMOTE"
