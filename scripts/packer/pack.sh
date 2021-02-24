#! /bin/bash
# Use Packer to produce an AWS AMI for autumn worker
set -e

if [ -f "pack.secret.sh" ]
then
    . ./pack.secret.sh
fi


if [[ -z "$SENTRY_DSN" ]]
then
    echo "SENTRY_DSN must be set for error reporting"
    exit 1
fi
if [[ -z "$AUTUMN_PASSWORD" ]]
then
    echo "AUTUMN_PASSWORD must be set for decrypting secrets"
    exit 1
fi
if [[ -z "$GCLOUD_API_KEY" ]]
then
    echo "GCLOUD_API_KEY must be set for Grafana log shipping."
    exit 1
fi
packer validate ami.json
packer build ami.json
