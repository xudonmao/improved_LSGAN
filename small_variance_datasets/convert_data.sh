#!/bin/bash

tar xzvf data.tar.gz
find data/times.shift -type f > times.shift.list
find data/times.shift.rotate -type f > times.shift.rotate.list
python convert_tfrecords.py times.shift.list times28_shift
python convert_tfrecords.py times.shift.rotate.list times28_shift_rotate
