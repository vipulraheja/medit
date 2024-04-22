#!/bin/sh

# Install Python3.9
sudo apt-get update
sudo apt-get install -y software-properties-common &&
sudo add-apt-repository -y ppa:deadsnakes/ppa

sudo apt-get install -y python3.9 python3.9-dev &&
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py &&
python3.9 get-pip.py &&
python3.9 -m pip install --upgrade pip

sudo apt install python3.9-venv

virtualenv -p /usr/bin/python3.9 venv
source venv/bin/activate
pip install -r requirements.txt
python3.9 -m unidic download
