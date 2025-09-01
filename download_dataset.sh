#!/usr/bin/env bash
set -eu

echo "Do u wanna leave the .zip files [y/n]"
read -r conf
while "$conf" != "y" && "$conf" != "n"; do
  echo "just [y/n] don't waste time"
  read -r conf
done

mkdir -p input
cd input

wget https://cvml.unige.ch/databases/DEAM/DEAM_Annotations.zip
wget https://cvml.unige.ch/databases/DEAM/DEAM_audio.zip
wget https://cvml.unige.ch/databases/DEAM/features.zip

for file in DEAM_Annotations.zip DEAM_audio.zip features.zip; do
  if [ -f "$file" ]; then
    unzip "$file" || echo "$file is corrupted"
    [ "$conf" == "y" ] && rm "$file"
  fi
done

cd ..
echo "Download complete!"
