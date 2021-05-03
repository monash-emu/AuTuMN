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
echo "GCLOUD_API_KEY=\"$GCLOUD_API_KEY\"" >> /etc/environment
echo "PYTHONUNBUFFERED=1" >> /etc/environment
cat /etc/environment

# Move uploaded files to the right place.
mv /tmp/promtail.yml /etc/promtail.yml
mv /tmp/promtail.service /etc/systemd/system/promtail.service
# Insert envar into promtail config.
sed -i "s/\${GCLOUD_API_KEY}/$GCLOUD_API_KEY/g" /etc/promtail.yml

# Set timezone to Melbourne
echo ">>> Setting timezone to Melbourne"
rm -f /etc/localtime
ln -sf /usr/share/zoneinfo/Australia/Melbourne /etc/localtime

# Setup public key so the Buildkite agent can SSH in.
BUILDKITE_PUBKEY="ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDCD/Li0eZR7WsrUWiolujjUEKcJgo5tErpcNAFcPxt0Ly7fLi5PGmx7RqP07W8myJPWMh/q4xFMBHqCCQrlqHvSOS+l8ExWYs6PzY/lCt721fFJRc16BbX3jJuvZlcYNK22IrMmjpvpKWS6kEqSWOufA4ZUEKpdSgSPZYVAQ9bivkQKS74uLdVmPVBdA56hzx2uo5UtrUqHnX1DFr40nmYEyGmZOUlUr3tp1quIXoapYNhfY0i3Xr4ivi8J3IaIqER93K8BTJh9+mjowh1TCBja9F8NwKD/AOQhMdIYtoj+QUGU8xfd0VsnFxLOl+LI7vYCqC1P7vQYpCBHqKcpOzN buildkite"
echo "$BUILDKITE_PUBKEY" >> /home/ubuntu/.ssh/authorized_keys

# Setup David's public key so he can SSH in.
DAVID_PUBKEY="ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC1VhP+uhhK2eKrkhkoagDMN4ySCmVMoQmPgK4tnz+0+vdyDE9Lo+n/rMqbTM2rBOcKl6OI/U5D+5FDgeQa4PDtBQB+9VogA+qmOdlo7U4qTHyX6sAl6+pHrwDS6WjJX6HEIZcl/3mtO3GQ1MtbJuDquzKQJn3u8JQEgi3R3GVRmyxZiNnVNxPAgBh7zN1qffBCmHmbQA6nnZY2Gj10SB7wrXLVrPduHcrmmp1TT2MVpC9wg5ImCWu2L+T3sSYwLQcX0x5awKpdM7kzi+BQh+pIUjH4yrEUr4YgygzTa9EBSUhz5izlAZW1fcjIjvYLp1VLttflr34zbDdbP33lRICb david.shipman@monash.edu"
echo "$DAVID_PUBKEY" >> /home/ubuntu/.ssh/authorized_keys

# Install basic requirements
apt-get update
apt-get install -y \
    python3-pip \
    python3.6-dev \
    virtualenv \
    unzip

# Install Git Large File Storage
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt install git-lfs

# Set up Grafana agent and Promtail log shipper
export GCLOUD_STACK_ID="174686"
GRAFANA_INSTALL_SCRIPT="https://raw.githubusercontent.com/grafana/agent/release/production/grafanacloud-install.sh"
curl -fsSL $GRAFANA_INSTALL_SCRIPT | sh

PROMTAIL_EXECUTABLE="https://github.com/grafana/loki/releases/download/v2.1.0/promtail-linux-amd64.zip"
curl -O -L  $PROMTAIL_EXECUTABLE
unzip promtail-linux-amd64.zip
mv promtail-linux-amd64 /usr/local/bin/promtail
chmod a+x /usr/local/bin/promtail

systemctl enable promtail.service

# Get source code
git clone https://github.com/monash-emu/AuTuMN.git code

# Install Python requirements
cd ~/code
virtualenv -p python3 env
. ./env/bin/activate
pip3 install -r requirements.txt
pip3 install awscli

# Give ownership of code folder to ubuntu user
sudo chown -R ubuntu:ubuntu ~/code
