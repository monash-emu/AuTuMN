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
SPRINGBOARD_LAUNCHKEY="ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDDt/yFdAAxB8zNEXSv80JYjTIK4j0XOxBFtYc0VixNpepib6BL7nOeaFWe22t5qcBhv0o+OfGWdpg7/lpsaz81maV2usoRIEg0R7nVesUEi8Bnua7rBfc4x27sSpEhxQ1IPUeotaDbY42EXqEmZhnZM/f18TUTAEUT4h0wOLJkTX778Eu/qYfSDYGnMyZ7uNMxreNaJrCt2/QmRVCptB1jpv9VubnshP/oYlPvWTZxuEFrN6ZjrgCtORzDiRm5gjTvGmLUxXEaZRv4mSQJuzLfXcw+V2hn86lPhdFzl88S/YhtaYcXw8Z3tgjir+C4hZGuazfjpYK+rBs34puSQP8bfHGTUelTRpyggf3GQ9fhGs8EGLaUlCkTGnr5Eq/EZkZKf677NHLP0HU8v0wh/HWKjxkfK/8W98CpKPSXB8VNGvbkekZALN++Bili6OeYX3HYo+pwrRxzULUqGoESw0iXbhE2LgZ8KhNL3iKRTi8VAUnIWOtU2XlL57UxpPyDZz8= springboard"
echo "$SPRINGBOARD_LAUNCHKEY" >> /home/ubuntu/.ssh/authorized_keys

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

# make awscli available without installing via conda
apt-get -y install awscli

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
