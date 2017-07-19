#!/bin/bash
# Assumes redis database is already running (check with "ps -ef | grep redis")
# Assumes the python flask server is running

cd `dirname $0`
cd ..
python -m celery -A server.tasks.celery_instance worker -l info
