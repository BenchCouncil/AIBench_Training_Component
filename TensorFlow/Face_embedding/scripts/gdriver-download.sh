#!/bin/bash -x

function gdrive_download () {
    CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
    wget -c --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
    rm -rf /tmp/cookies.txt
}

# CASIA-Webface
# file_id='1Of_EVz-yHV7QVWQGihYfvtny9Ne8qXVz'
# file_name='CASIA-Webface.zip'
# gdrive_download ${file_id} ${file_name}

# Model name    LFW accuracy    Training dataset    Architecture
# 20180408-102900 0.9905  CASIA-WebFace   Inception ResNet v1
# 20180402-114759 0.9965  VGGFace2    Inception ResNet v1
# Inception-ResNet for CASIA-Webface
# file_id='1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz'
# file_name='20180408-102900.zip'
# gdrive_download ${file_id} ${file_name}
# Inception-ResNet for VGGFace2
# file_id='1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-'
# file_name='20180402-114759.zip'

file_id=$1
file_name=$2
gdrive_download ${file_id} ${file_name}
