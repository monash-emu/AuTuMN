#!/bin/bash
# Builds and deploys website
set -e
SCRIPT_DIR=$(dirname $0)
pushd $SCRIPT_DIR
if [ ! -d "../aws/env" ]
then
    echo "Installing Python requirements"
    virtualenv --quiet -p python3 ../aws/env
    . ../aws/env/bin/activate
    pip3 install --quiet boto3 awscli
fi
. ../aws/env/bin/activate

echo "Fetching website data"
./read_website_data.py

echo "Installing JavaScript requirements"
yarn install

echo "Building web assets"
rm -rf out
yarn build

echo "Uploading web assets"
if [[ -d ~/.aws/ ]]
then
    aws --profile autumn s3 cp --recursive out s3://www.autumn-data.com
else
    aws s3 cp --recursive out s3://www.autumn-data.com
fi
echo "Done uploading website"
popd
