#!/usr/bin/env bash

# read $NB_CAPT
echo "CP, FD or ITW?"
read $FILE
export var=$FILE
cat << EOF > run_project.py
#!/usr/bin/python
import subprocess
from data_mining_project import main
main(files=subprocess.call(["echo","var"]))
EOF
chmod 755 run_project.py
./run_project.py

# FIXME take off the error in the output
# main(files=subprocess.call(["echo", "$FILE"])
