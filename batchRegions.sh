#!/usr/bin/env bash

subjects="0110171 0110172 0111171"
hemis="lh rh"

roiloc=wang2014
for sub in $subjects;
do
  for hemi in $hemis;
  do
    for num in {1..25};
    do
      sbatch classRegion.sh $sub neurosketch $roiloc $num $hemi
      echo $sub neurosketch $roiloc $num $hemi
    done
  done
done



roiloc=schaefer2018
for sub in $subjects;
do
  for num in {1..300};
  do
    sbatch classRegion.sh $sub neurosketch $roiloc $num
    echo $sub neurosketch $roiloc $num
  done
done
