#!/usr/bin/env bash
# subjects="1206161 0119173 1206162 1201161 0115174 1130161 1206163 0120171 0111171 1202161 1121161 0125172 0110172 0123173 0120172 0113171 0115172 0120173 0110171 0119172 0124171 0123171 1203161 0118172 0118171 0112171 1207162 0119171 0117171 0119174 0112173 0112174 0125171 0112172" # al subjects
subjects="1206161 0119173 1206162 1201161 1130161 1206163 0120171 0111171 1202161 1121161 0125172 0110172 0123173 0115172 0120173 0110171 0119172 0124171 0123171 1203161 0118172 0118171 0112171 1207162 0117171 0119174 0112173 0112174 0112172" # good align subjects
for sub in $subjects;do
    sbatch make-schaefer-rois.sh $sub
done