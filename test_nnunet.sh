#!/bin/bash

input_dir=$1
output_dir=$2

if [ ! -d "$output_dir"/imagesTr ]; then
    mkdir -p "$output_dir"/imagesTr
fi

if [ ! -d "$output_dir"/labelsTr ]; then
    mkdir -p "$output_dir"/labelsTr
fi

for chanel0 in "$input_dir"/*_LF.nii.gz; do
    basenamed=$(basename $chanel0)
    cp "$chanel0" "$output_dir"/imagesTr/"${basenamed/_LF/_0000}"
done

for chanel1 in "$input_dir"/*_SR.nii.gz; do
    basenamed=$(basename $chanel1)
    cp "$chanel1" "$output_dir"/imagesTr/"${basenamed/_SR/_0001}"
done

for gt in "$input_dir"/*_GT.nii.gz; do
    basenamed=$(basename $gt)
    cp "$gt" "$output_dir"/labelsTr/"${basenamed/_GT/}"
done


# create json file
dataset_json="$output_dir"/dataset.json
if [ -e $dataset_json ]; then
  rm "$dataset_json"
fi

cat > $dataset_json <<EOF
{
 "channel_names": {
   "0": "MRI",
   "1": "MRI"
 },
 "labels": {
   "background": 0,
   "LH": 1,
   "RH": 2,
   "LV": 3,
   "RV": 4,
   "LC": 5,
   "RC": 6,
   "LL": 7,
   "RL": 8
 },
 "numTraining": $(ls "$input_dir"/*_GT.nii.gz | wc -l | xargs),
 "file_ending": ".nii.gz"
}
EOF

# now plan!
export nnUNet_raw="$(dirname "$output_dir")"
export nnUNet_preprocessed="$(dirname "$(dirname "$output_dir")")"
export nnUNet_results=""
nnUNetv2_plan_and_preprocess -d 003 --verify_dataset_integrity --no_pp --clean -c 3d_fullres --verbose
nnUNetv2_plan_and_preprocess -d 003 --no_pp --clean -c 3d_fullres --verbose -pl nnUNetPlannerResEncM
nnUNetv2_plan_and_preprocess -d 003 --no_pp --clean -c 3d_fullres --verbose -pl nnUNetPlannerResEncL
nnUNetv2_plan_and_preprocess -d 003 --no_pp --clean -c 3d_fullres --verbose -pl nnUNetPlannerResEncL -overwrite_plans_name ResEncL12GB -gpu_memory_target 12