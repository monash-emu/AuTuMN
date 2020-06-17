#! /bin/bash
# Use Packer to produce an AWS AMI for autumn worker
set -e
if [[ -z "$SENTRY_DSN" ]]
then
    echo "SENTRY_DSN must be set"
    exit 1
fi
packer validate ami.json
packer build ami.json