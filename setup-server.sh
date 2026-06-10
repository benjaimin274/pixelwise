#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pull the pinned model artefact
# CHANGE: Added $SCRIPT_DIR to find .env correctly from anywhere
if [ -f "$SCRIPT_DIR/.env" ]; then
set -a; source "$SCRIPT_DIR/.env"; set +a
if [ -n "${MODEL_REPO:-}" ] && \
[ -n "${MODEL_VERSION:-}" ]; then
mkdir -p models/
rm -rf /tmp/pixelwise-model
git clone --depth 1 --branch "$MODEL_VERSION" \
"$MODEL_REPO" /tmp/pixelwise-model
cp /tmp/pixelwise-model/*.pkl models/
cp /tmp/pixelwise-model/MODELCARD.md models/
rm -rf /tmp/pixelwise-model
fi
fi

# 2. Sync Python virtual environment requirements
# CHANGE: Added $SCRIPT_DIR to track requirements.txt path safely
if [ -d .venv ] && [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    echo "Virtual environment and requirements list found. Syncing dependencies..."
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r "$SCRIPT_DIR/requirements.txt"
else
    echo "Warning: .venv directory or requirements.txt missing. Skipping dependency installation."
fi

# Install, start, and report the systemd unit on prod
if [ -f deploy/pixelwise.service ] && \
   command -v systemctl >/dev/null 2>&1 && \
   id produser >/dev/null 2>&1; then
    echo "Production environment detected. Installing systemd service..."
    sudo cp deploy/pixelwise.service /etc/systemd/system/pixelwise.service
    sudo systemctl daemon-reload
    sudo systemctl enable pixelwise
    sudo systemctl restart pixelwise
    sudo systemctl status pixelwise --no-pager
fi

# ADDITION: Paste the new PostgreSQL provisioning block right here at the bottom
# Provision the pixelwise role and database on every VM
if command -v psql >/dev/null 2>&1 && \
[ -f "$SCRIPT_DIR/.env" ]; then
set -a; source "$SCRIPT_DIR/.env"; set +a
sudo -u postgres psql -tAc \
"SELECT 1 FROM pg_roles WHERE rolname='pixelwise'" \
| grep -q 1 || \
sudo -u postgres psql -c \
"CREATE USER pixelwise \
WITH PASSWORD '$DB_PASSWORD';"
sudo -u postgres psql -tAc \
"SELECT 1 FROM pg_database WHERE datname='pixelwise'" \
| grep -q 1 || \
sudo -u postgres createdb -O pixelwise pixelwise
fi