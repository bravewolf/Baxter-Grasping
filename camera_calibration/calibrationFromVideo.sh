#!/bin/sh
echo "Enter video name"
read VIDEO
./ca "calibrate.txt" -dp="detector_params.yml" -v="$VIDEO" -w=5 -h=7 -sl=0.034 -ml=0.02 -d=10


