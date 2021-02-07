#!/usr/bin/env bash

# subjects="1206161 0119173 1206162 1201161 1130161 1206163 0120171 0111171 1202161 1121161 0125172 0110172 0123173 0115172 0120173 0110171 0119172 0124171 0123171 1203161 0118172 0118171 0112171 1207162 0117171 0119174 0112173 0112174 0112172"
subjects="1206161 0119173 1206162 1201161 1130161 1206163 0120171 0111171 1202161 1121161 0125172 0110172 0123173 0115172 0120173 0110171 0119172 0124171 0123171 1203161 0118172 0118171 0112171 1207162 0117171 0119174 0112173 0112174 0112172" # good align subjects
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
