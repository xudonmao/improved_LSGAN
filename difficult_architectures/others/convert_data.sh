#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 data_directory"
  exit 1
fi

find $1  -type f > file.list
python convert_tfrecords.py file.list bedroom64

