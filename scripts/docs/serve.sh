#!/bin/bash
SCRIPT_DIR=$(dirname $0)
DOCS_DIR=`realpath $SCRIPT_DIR/../../docs`
pushd "$DOCS_DIR/_build/html"
    python -m http.server
popd
