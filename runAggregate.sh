#!/usr/bin/env bash

# The purpose of this code is to run code like this `sbatch aggregate.sh 0111171 neurosketch schaefer2018 15` for all 3 sub and all number of ROIs
# and then load the accuracy result of this code and compare them.

# subjects="0110171 0110172 0111171"
subjects="1206161 0119173 1206162 1201161 0115174 1130161 1206163 0120171 0111171 1202161 1121161 0125172 0110172 0123173 0120172 0113171 0115172 0120173 0110171 0119172 0124171 0123171 1203161 0118172 0118171 0112171 1207162 0119171 0117171 0119174 0112173 0112174 0125171 0112172"
hemis="lh rh"

roiloc=wang2014
for sub in $subjects;
do
  for num in {1..50};
  do
      sbatch aggregate.sh $sub neurosketch $roiloc $num
      echo $sub neurosketch $roiloc $num
  done
done



roiloc=schaefer2018
for sub in $subjects;
do
  for num in {1..300};
  do
    sbatch aggregate.sh $sub neurosketch $roiloc $num
    echo $sub neurosketch $roiloc $num
  done
done


