#!/bin/bash

seeds=(9741 82936 47261 11665)
gpu=0

for seed in "${seeds[@]}"; do
    python train-baseline.py $seed ${gpu}

done
