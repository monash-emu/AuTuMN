#!/bin/bash
# On MacOSX, checks whether redis and postgres is running
# Version: 2016nov15

if [[ $(lunchy status postgres | grep postgres) ]]; then
    echo "Postgres already running..."
else
    echo "Start postgres"
    lunchy start postgres
fi

if [[ $(lunchy status redis | grep redis) ]]; then
    echo "Redis already running..."
else
    echo "Start redis"
    lunchy start redis
fi
