# Mask persons

Python script to mask persons in an image by use of a neuronal network.

## Installation

### Prerequisites

Make shure you have python3, python3-venv and python3-pip installed. 

```
sudo apt upgrade
sudo apt install python3 -y
sudo apt install python3-venv -y
sudo apt install python3-pip -y
```

### Virtual python environment

To be independent from the installed python3 modules you have to create a python virtual environment inside global dir `/usr/local/bin`. Do this as root. To become root execute `su -` or `sudo su -`. Later can use this virtual environment as the shebang inside the python script.

```
ENV_DIR='/usr/local/bin/env'
mkdir -p ${ENV_DIR}
python3 -m venv ${ENV_DIR} >/dev/null 2>&1
source ${ENV_DIR}/bin/activate
pip install opencv-python
pip install imutils
deactivate
```

To use exactly this python environment `#!/usr/local/bin/env/bin/python3`

### Install the Script

The script uses a neuronal network. This network ist saved inside a subdirectory. Go to the directory where you want to store the script and clone it from github:

```
git clone https://github.com/mietkamera/${GIT_REPO_NAME}
```
