# Descriptive script that can be used to set up our Buildkite server. 
apt update
apt upgrade -y
# Install Python dependencies.
apt-get install -qq python3-pip virtualenv

# Install Yarn package manager.
curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add -
echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list
apt-get update && apt-get install yarn -qq

# Install Node Version Manager + latest NodeJS.
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.36.0/install.sh | bash
# Exit shell and re-open
nvm install node

python3 -V
pip3 -V
node --version
yarn --version

# See https://buildkite.com/organizations/autumn/agents#setup-ubuntu
# Run as root
AGENT_TOKEN=xxx
sh -c 'echo deb https://apt.buildkite.com/buildkite-agent stable main > /etc/apt/sources.list.d/buildkite-agent.list'
apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 32A37959C2FA5C3C99EFBC32A79206696452D198
apt-get update
apt-get install -y buildkite-agent
sed -i "s/xxx/$AGENT_TOKEN/g" /etc/buildkite-agent/buildkite-agent.cfg
head /etc/buildkite-agent/buildkite-agent.cfg

# Add SSH keys
cd /var/lib/buildkite-agent/
mkdir .ssh
nano ./.ssh/id_rsa.pub
nano ./.ssh/id_rsa
chmod 400 ./.ssh/id_rsa
chown -R buildkite-agent:buildkite-agent ./.ssh
ls -la /var/lib/buildkite-agent/.ssh

# Add AWS creds to /etc/buildkite-agent/hooks/environment
nano /etc/buildkite-agent/hooks/environment
chmod +x /etc/buildkite-agent/hooks/environment
chown -R buildkite-agent:buildkite-agent /etc/buildkite-agent/hooks/

# Create a systemd template
cp /lib/systemd/system/buildkite-agent.service /etc/systemd/system/buildkite-agent@.service

# Start 12 agents using the systemd template we created above
systemctl enable --now buildkite-agent@1
systemctl enable --now buildkite-agent@2
systemctl enable --now buildkite-agent@3
systemctl enable --now buildkite-agent@4
systemctl enable --now buildkite-agent@5
systemctl enable --now buildkite-agent@6
