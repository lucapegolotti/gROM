#!/bin/bash

export OMP_NUM_THREADS=3

if [ "$#" -ne 2 ]; then
    echo "Usage: arg#1 = experiment id, arg#2 = num workers"
    exit 1
fi

export EXPCODE=$1
export NUMWORKERS=$2

for (( i=0; i<$NUMWORKERS; i++ ))
do
    ( sigopt start-worker $EXPCODE python training_sigopt.py > $i.txt & )
done
