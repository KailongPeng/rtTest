#!/usr/bin/env bash
# Input python command to be submitted as a job

#SBATCH --output=logs/maskmaker-%j.out
#SBATCH --job-name flirt
#SBATCH --partition=short,scavenge
#SBATCH --time=1:00:00
#SBATCH --mem=10000
#SBATCH -n 1

module load FSL/5.0.9

sub=$1 
num=$2
roiloc=$3
STAND=/gpfs/milgram/apps/hpc.rhel7/software/FSL/5.0.10-centos7_64/data/standard/MNI152_T1_1mm_brain.nii.gz
if [roiloc = wang2014]
then     
    # convert the topMasks into standard space
        # convert wang2func.mat to func2wang.mat
    WANG2FUNC=./wang2014/${sub}/wang2func.mat
    FUNC2WANG=./wang2014/${sub}/func2wang.mat
    convert_xfm -omat ${FUNC2WANG} -inverse ${WANG2FUNC}
        # convert individual space to stand space
    INPUT=./wang2014/${sub}/output/top${num}mask.nii.gz
    OUTPUT=./wang2014/${sub}/output/STAND_top${num}mask.nii.gz
    echo flirt -ref $STAND -in $INPUT -out $OUTPUT -applyxfm -init $FUNC2WANG
    flirt -ref $STAND -in $INPUT -out $OUTPUT -applyxfm -init $FUNC2WANG
else
    FUNC2WANG=./wang2014/${sub}/func2wang.mat
    INPUT=./schaefer2018/${sub}/output/top${num}mask.nii.gz
    OUTPUT=./schaefer2018/${sub}/output/STAND_top${num}mask.nii.gz
    echo flirt -ref $STAND -in $INPUT -out $OUTPUT -applyxfm -init $FUNC2WANG
    flirt -ref $STAND -in $INPUT -out $OUTPUT -applyxfm -init $FUNC2WANG
fi
