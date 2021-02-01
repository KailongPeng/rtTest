#!/usr/bin/env bash

subjects="0110171 0110172 0111171"

for sub in $subjects;
do
  sbatch assemble2.sh $sub
done
