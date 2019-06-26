#!/usr/bin/env bash

echo "CP, FD or ITW?"
read F
if [ -f out.txt ]; then
   rm out.txt
   echo "out.txt updated"
fi
python ./data_mining_project.py $F >> out.txt
