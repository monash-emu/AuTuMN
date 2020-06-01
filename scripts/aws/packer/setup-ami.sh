#! /bin/bash
# Build AWS deep learning machine image.
# Called by Packer to produce an AWS AMI
set -e

# Handle some magical apt sources race condition bullshit.
echo "Waiting up to 180 seconds for cloud-init to update /etc/apt/sources.list"
timeout 180 /bin/bash -c 'until stat /var/lib/cloud/instance/boot-finished 2>/dev/null; do echo waiting ...; sleep 1; done'

# Install basic requirements
sudo apt-get update
sudo apt-get install -y \
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
