#!/usr/bin/env bash
set -e
mkdir -p input
cd input

wget https://cvml.unige.ch/databases/DEAM/DEAM_Annotations.zip
wget https://cvml.unige.ch/databases/DEAM/DEAM_audio.zip
wget https://cvml.unige.ch/databases/DEAM/features.zip

cd ..
echo "Download complete!"
