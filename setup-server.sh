#!/bin/bash
sudo apt install -y python3-venv python3-pip postgresql postgresql-contrib nginx

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pull the pinned model artefact
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a; source "$SCRIPT_DIR/.env"; set +a
    if [ -n "${MODEL_REPO:-}" ] && [ -n "${MODEL_VERSION:-}" ]; then
        mkdir -p "$SCRIPT_DIR/models/"
        rm -rf /tmp/pixelwise-model
        git clone --depth 1 --branch "$MODEL_VERSION" "$MODEL_REPO" /tmp/pixelwise-model
        cp /tmp/pixelwise-model/*.pkl "$SCRIPT_DIR/models/"
        cp /tmp/pixelwise-model/MODELCARD.md "$SCRIPT_DIR/models/"
        rm -rf /tmp/pixelwise-model
    fi
fi

# Sync Python virtual environment requirements & execute training pipelines
if [ -d "$SCRIPT_DIR/.venv" ] && [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    echo "Virtual environment and requirements list found. Syncing dependencies..."
    "$SCRIPT_DIR/.venv/bin/pip" install --upgrade pip
    "$SCRIPT_DIR/.venv/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"

    if [ -f "$SCRIPT_DIR/OOD detection/train_isolation_forest.py" ]; then
        echo "ML Pipeline: Executing OOD Isolation Forest training sequence..."
        "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/OOD detection/train_isolation_forest.py"
        mkdir -p "$SCRIPT_DIR/models"
        echo "ML Pipeline: Training sequence complete. Core artifact materialized."
    fi
else
    echo "Warning: .venv directory or requirements.txt missing. Skipping dependency installation."
fi

# Install, start, and report the systemd unit on prod
if [ -f "$SCRIPT_DIR/deploy/pixelwise.service" ] && \
   command -v systemctl >/dev/null 2>&1 && \
   id produser >/dev/null 2>&1; then
    echo "Production environment detected. Installing systemd service..."
    sudo cp "$SCRIPT_DIR/deploy/pixelwise.service" /etc/systemd/system/pixelwise.service
    sudo systemctl daemon-reload
    sudo systemctl enable pixelwise
    sudo systemctl restart pixelwise
    sudo systemctl status pixelwise --no-pager
fi

# Provision the pixelwise role and database on every VM
if command -v psql >/dev/null 2>&1 && [ -f "$SCRIPT_DIR/.env" ]; then
    set -a; source "$SCRIPT_DIR/.env"; set +a
    sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='pixelwise'" | grep -q 1 || \
    sudo -u postgres psql -c "CREATE USER pixelwise WITH PASSWORD '$DB_PASSWORD';"
    
    sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='pixelwise'" | grep -q 1 || \
    sudo -u postgres createdb -O pixelwise pixelwise
fi

# Install Nginx site and deploy the frontend on prod only
if [ -f "$SCRIPT_DIR/deploy/pixelwise.nginx" ] && \
   command -v nginx >/dev/null 2>&1 && \
   id produser >/dev/null 2>&1; then

    echo "Edge Provisioning: Materializing frontend file structure..."
    sudo mkdir -p /var/www/pixelwise
    sudo cp -r "$SCRIPT_DIR/frontend/"* /var/www/pixelwise/

    echo "Edge Provisioning: Injecting active SECRET_API_KEY from .env..."
    KEY=$(grep ^SECRET_API_KEY "$SCRIPT_DIR/.env" | cut -d'=' -f2)
    sudo sed -i "s/REPLACE_ME/$KEY/" /var/www/pixelwise/app.js

    echo "Edge Provisioning: Staging Nginx configuration files..."
    sudo cp "$SCRIPT_DIR/deploy/pixelwise.nginx" /etc/nginx/sites-available/pixelwise
    sudo ln -sf /etc/nginx/sites-available/pixelwise /etc/nginx/sites-enabled/pixelwise

    echo "Edge Provisioning: Cleaning up default configuration blockers..."
    sudo rm -f /etc/nginx/sites-enabled/default

    echo "Edge Provisioning: Testing syntax rules and reloading daemon..."
    sudo nginx -t && sudo systemctl reload nginx
    echo "Edge Provisioning: Frontend application successfully deployed at port 80!"
fi

# Install and initialize the auto-deploy systemd scheduling timer on prod
if [ -f "$SCRIPT_DIR/deploy/systemd/pixelwise-deploy.timer" ] && \
   command -v systemctl >/dev/null 2>&1 && \
   id produser >/dev/null 2>&1; then

    echo "Pipeline Setup: Staging automated deployment configuration units..."
    sudo cp "$SCRIPT_DIR/deploy/systemd/pixelwise-deploy.service" /etc/systemd/system/pixelwise-deploy.service
    sudo cp "$SCRIPT_DIR/deploy/systemd/pixelwise-deploy.timer" /etc/systemd/system/pixelwise-deploy.timer

    echo "Pipeline Setup: Granting produser passwordless restart permissions..."
    echo "produser ALL=(root) NOPASSWD: /usr/bin/systemctl restart pixelwise" | sudo tee /etc/sudoers.d/pixelwise > /dev/null

    echo "Pipeline Setup: Reloading background system engines and activating scheduler..."
    sudo systemctl daemon-reload
    sudo systemctl enable --now pixelwise-deploy.timer
    echo "Pipeline Setup: Automation loop initialized! System triggers every 30 seconds."
fi