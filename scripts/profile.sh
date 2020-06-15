#!/bin/bash
# Runs a script called profile.secret.py in the project root and profiles performance using cProfile
# See this blog for more details https://julien.danjou.info/guide-to-python-profiling-cprofile-concrete-case-carbonara/
# Requires KCachegrind https://kcachegrind.github.io/html/Home.html
python -m cProfile -o data/profile-fixed.cprof profile.secret.py
pyprof2calltree -k -i data/profile-fixed.cprof
