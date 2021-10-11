# Mask persons

Python script to mask persons in an image by use of a neuronal network.

## Installation

Make shure you have python3 and python3-pip installed. 

```
sudo apt upgrade
sudo apt install python3 -y
sudo apt install python3-venv -y
sudo apt install python3-pip -y
```

Create an python virtual environment inside global dir `/usr/local/bin`. Do this as root. To
become root execute `su -` or `sudo su -`.

```
ENV_DIR='/usr/local/bin/env'
GIT_REPO_NAME='mask_persons'  
mkdir -p ${ENV_DIR}
python3 -m venv ${ENV_DIR} >/dev/null 2>&1
source ${ENV_DIR}/bin/activate
pip install opencv-python
pip install imutils
deactivate
GIT_REPO_NAME='mask_persons'  
cd ${ENV_DIR}                   
git clone https://github.com/mietkamera/${GIT_REPO_NAME}
chmod -R 777 ${ENV_DIR}
chgrp -R users ${ENV_DIR}/${GIT_REPO_NAME}
```


