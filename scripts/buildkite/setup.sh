# See https://buildkite.com/organizations/autumn/agents#setup-ubuntu
# Run as root
# TODO: Put all SSH keys in /var/lib/buildkite-agent/.ssh/
# TODO: Add AWS creds to /etc/buildkite-agent/hooks/environment
AGENT_TOKEN=xxx
sh -c 'echo deb https://apt.buildkite.com/buildkite-agent stable main > /etc/apt/sources.list.d/buildkite-agent.list'
apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 32A37959C2FA5C3C99EFBC32A79206696452D198
apt-get update
apt-get install -y buildkite-agent
sed -i "s/xxx/$AGENT_TOKEN/g" /etc/buildkite-agent/buildkite-agent.cfg
systemctl enable buildkite-agent
systemctl start buildkite-agent
# journalctl -f -u buildkite-agent

# Upgrade to 2 agents
systemctl stop buildkite-agent
systemctl disable buildkite-agent

# Create a systemd template
cp /lib/systemd/system/buildkite-agent.service /etc/systemd/system/buildkite-agent@.service

# Start 9 agents using the systemd template we created above
systemctl enable --now buildkite-agent@1
systemctl enable --now buildkite-agent@2
systemctl enable --now buildkite-agent@3
systemctl enable --now buildkite-agent@4
systemctl enable --now buildkite-agent@5
systemctl enable --now buildkite-agent@6
systemctl enable --now buildkite-agent@7
systemctl enable --now buildkite-agent@8
systemctl enable --now buildkite-agent@9

apt-get install -qq python3-pip virtualenv

# If you need to disable agents
systemctl disable buildkite-agent@1
systemctl disable buildkite-agent@2
systemctl disable buildkite-agent@3
systemctl disable buildkite-agent@4
systemctl disable buildkite-agent@5
systemctl disable buildkite-agent@6
systemctl disable buildkite-agent@7
systemctl disable buildkite-agent@8
systemctl disable buildkite-agent@9


