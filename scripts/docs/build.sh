#!/bin/bash
set -e
SCRIPT_DIR=$(dirname $0)
DOCS_DIR=`realpath $SCRIPT_DIR/../../docs`

echo "Building summer examples"
pushd $SCRIPT_DIR
    python build_examples.py
popd

echo -e "\nBuilding documentation HTML"
pushd $DOCS_DIR
    rm -rf ./_build/doctrees  ./_build/html/*
    make html
popd
echo -e "\nFinished building documentation HTML"
