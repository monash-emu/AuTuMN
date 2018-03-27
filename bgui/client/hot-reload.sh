#!/usr/bin/env bash
if [ ! -f src/config.js ]; then
    cp src/defaultConfig.js src/config.js
fi
npm run dev
