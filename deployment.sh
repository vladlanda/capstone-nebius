#!/bin/bash
set -e

# ==============================
# CONFIG (edit if needed)
# ==============================
APP_DIR="$(pwd)"
APP_FILE="server.py"

SERVICE_NAME="streamlit"
USER_NAME="$(whoami)"

STREAMLIT_PORT="8501"
PUBLIC_PORT="80"

SERVER_MAX_UPLOAD_FILE_MB="1000"
# ==============================

echo "🚀 Deploying Streamlit with uv + systemd + Nginx..."
echo "📍 App directory: $APP_DIR"

# ------------------------------
# 1. Install system packages
# ------------------------------
echo "📦 Installing system dependencies..."
sudo apt update
sudo apt install -y nginx curl

# ------------------------------
# 2. Install uv if missing
# ------------------------------
if ! command -v uv &> /dev/null; then
  echo "⚡ Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# ------------------------------
# 3. Create virtual environment
# ------------------------------
echo "🐍 Creating .venv..."
uv venv

# ------------------------------
# 4. Install Python requirements
# ------------------------------
echo "📦 Installing requirements..."
uv pip install -r requirements.txt

# ------------------------------
# 5. Create systemd service
# ------------------------------
echo "🛠 Setting up systemd service..."

SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

sudo tee $SERVICE_FILE > /dev/null <<EOF
[Unit]
Description=Streamlit App
After=network.target

[Service]
User=${USER_NAME}
WorkingDirectory=${APP_DIR}

ExecStart=${APP_DIR}/.venv/bin/python -m streamlit run ${APP_FILE} \
  --server.address 0.0.0.0 \
  --server.port ${STREAMLIT_PORT} \
  --server.maxUploadSize ${SERVER_MAX_UPLOAD_FILE_MB}

Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# Enable + start Streamlit
sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}
sudo systemctl restart ${SERVICE_NAME}

# ------------------------------
# 6. Configure Nginx reverse proxy
# ------------------------------
echo "🌍 Configuring Nginx for port ${PUBLIC_PORT}..."

NGINX_CONF="/etc/nginx/sites-available/${SERVICE_NAME}"

sudo tee $NGINX_CONF > /dev/null <<EOF
server {
    listen ${PUBLIC_PORT};
    server_name _;
    client_max_body_size ${SERVER_MAX_UPLOAD_FILE_MB}M;
    location / {
        proxy_pass http://127.0.0.1:${STREAMLIT_PORT};

        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
    }
}
EOF

# Enable site + remove default
sudo ln -sf $NGINX_CONF /etc/nginx/sites-enabled/${SERVICE_NAME}
sudo rm -f /etc/nginx/sites-enabled/default

# Restart nginx
sudo systemctl restart nginx
sudo systemctl enable nginx

# ------------------------------
# Done
# ------------------------------
echo ""
echo "🎉 Deployment complete!"
echo ""
echo "Streamlit is running internally on: ${STREAMLIT_PORT}"
echo "Nginx is exposing it publicly on:  http://<VM_PUBLIC_IP>/"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status ${SERVICE_NAME}"
echo "  journalctl -u ${SERVICE_NAME} -f"
