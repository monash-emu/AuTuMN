#! /bin/bash
# Build AWS deep learning machine image.
# Called by Packer to produce an AWS AMI
set -e

# Handle some magical apt sources race condition bullshit.
echo ">>> Waiting up to 180 seconds for cloud-init to update /etc/apt/sources.list"
timeout 180 /bin/bash -c 'until stat /var/lib/cloud/instance/boot-finished 2>/dev/null; do echo waiting ...; sleep 1; done'

echo ">>> Setting global environment variables"
echo "SENTRY_DSN=\"$SENTRY_DSN\"" >> /etc/environment
echo "AUTUMN_PASSWORD=\"$AUTUMN_PASSWORD\"" >> /etc/environment
echo "PYTHONUNBUFFERED=1" >> /etc/environment
cat /etc/environment

# Set timezone to Melbourne
echo ">>> Setting timezone to Melbourne"
rm -f /etc/localtime
ln -sf /usr/share/zoneinfo/Australia/Melbourne /etc/localtime

# Setup public key for buildkite
BUILDKITE_PUBKEY="ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDCD/Li0eZR7WsrUWiolujjUEKcJgo5tErpcNAFcPxt0Ly7fLi5PGmx7RqP07W8myJPWMh/q4xFMBHqCCQrlqHvSOS+l8ExWYs6PzY/lCt721fFJRc16BbX3jJuvZlcYNK22IrMmjpvpKWS6kEqSWOufA4ZUEKpdSgSPZYVAQ9bivkQKS74uLdVmPVBdA56hzx2uo5UtrUqHnX1DFr40nmYEyGmZOUlUr3tp1quIXoapYNhfY0i3Xr4ivi8J3IaIqER93K8BTJh9+mjowh1TCBja9F8NwKD/AOQhMdIYtoj+QUGU8xfd0VsnFxLOl+LI7vYCqC1P7vQYpCBHqKcpOzN buildkite"
echo "$BUILDKITE_PUBKEY" >> /home/ubuntu/.ssh/authorized_keys

# Install basic requirements
apt-get update
apt-get install -y \
    python3-pip \
    python3.6-dev \
    virtualenv


# Get source code
git clone https://github.com/monash-emu/AuTuMN.git code

# Install Python requirements
cd ~/code
virtualenv -p python3 env
. ./env/bin/activate
pip3 install -r requirements.txt
pip3 install awscli
