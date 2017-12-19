#!/bin/bash

sh setting_up_script.sh
find cats_bigger_than_128x128 -type f > file.list
python convert_tfrecords.py file.list cat_128
