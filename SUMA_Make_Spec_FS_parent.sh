SUMA_Make_Spec_FS_parent.sh

subjects="1206161 0119173 1206162 1130161 1206163 0120171 0111171 1202161 0125172 0110172 0123173 0120173 0110171 0119172 0124171 0123171 1203161 0118172 0118171 0112171 1207162 0117171 0119174 0112173 0112172" #these subjects are done with the batchRegions code
hemis="lh rh"
roiloc=wang2014
for sub in $subjects;do
    sbatch SUMA_Make_Spec_FS.sh $sub
    echo sbatch SUMA_Make_Spec_FS.sh $sub
done