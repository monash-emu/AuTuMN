#! /bin/bash
# Build AWS deep learning machine image.
# Called by Packer to produce an AWS AMI
set -e

# Handle some magical apt sources race condition bullshit.
echo ">>> Waiting up to 180 seconds for cloud-init to update /etc/apt/sources.list"
timeout 180 /bin/bash -c 'until stat /var/lib/cloud/instance/boot-finished 2>/dev/null; do echo waiting ...; sleep 1; done'

echo ">>> Setting global environment variables"
echo "AUTUMN_PASSWORD=\"$AUTUMN_PASSWORD\"" >> /etc/environment
echo "PYTHONUNBUFFERED=1" >> /etc/environment
cat /etc/environment

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

# Setup the EMU group launchkey
EMU_LAUNCHKEY="ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDLuxwgC84Qy7flUMxpHe8crkAh3z9DQ80bAaVK+GuCcGODG+lBc4wlnmtCUQnDWqdiYKwqqBGWwTv1H1AOqE7TjiC51RjAzkPQEkN18azbnyP6u4WcVw3SdUVVKA/MpmfR5GAY70TfyeoAjb4uJf5hUDKhA586vObtyP41/OAVP2scnjf7KyDk0buuSr2HVt1mQE4BBwqoUmZhcyyPrQIb+sfyHBfwzzkZFOJWDikARm8q/ncVkRWEXw+3lRmBvign23Onv8FxdU2XCKAe2gSUGgdDky6tvDtTVDHDT9noY78Sg1q/tZ+uDj2+crse+EvBAmS2QBQeV04etdaRy8R26CumjRd1rzL0wgtBrquNTLXNfZWSeitKZ4pEHqDI2AwmMj6TEjKbS+uvMJvJfoLqM9KHUxwP26iCPDhCCgKZteZUo+/wLsXxUYSuLSVue6SpXy2XdybZZt/eaMwxaa+A0GyeTAZo0UDZfNFlPCqtalQEPHwIXnO7JqFHY4BA2Ss= emu-launchkey"
echo "$EMU_LAUNCHKEY" >> /home/ubuntu/.ssh/authorized_keys

# Install basic requirements
apt-get update
apt-get -y upgrade

# Install Git Large File Storage
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get -y install git-lfs

# Do our other apt installs
# g++ (for fast PyMC)
apt-get -y install g++

# openblas (as above for PyMC)
apt-get -y install libopenblas-dev

# libgraphviz - for interactive computegraph
apt-get -y install libgraphviz-dev

# tex things - not really a core requirement but takes forever to install so may as well...
apt-get -y install texlive-latex-base texlive-fonts-recommended texlive-fonts-extra texlive-latex-extra texlive-bibtex-extra biber

# Install miniconda and set up hooks

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda
sudo chown -R ubuntu:ubuntu ~/miniconda

rm ~/miniconda.sh

eval "$(~/miniconda/bin/conda shell.bash hook)"
conda init bash

source ~/.bashrc

mkdir ~/code

# Install Python requirements
cd ~/code

# Get source code
git clone https://github.com/monash-emu/AuTuMN.git autumn

cd autumn

git pull

conda create -y -n autumn310 python=3.10
conda activate autumn310

pip3 install -r requirements/requirements310.txt
pip3 install -e ./
conda deactivate

# Give ownership of code folder to ubuntu user
sudo chown -R ubuntu:ubuntu ~/code
