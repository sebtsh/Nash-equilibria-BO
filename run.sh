echo "exp_type: $1"
echo "utility_name: $2";
echo "acq_name: $3";

mkdir out

if [[ "$1" == "pne" ]]
then
  for i in {1..4}
  do
    echo "Running PNE with ""$2"" acq_name=""$3"" seed=""$i"""
    CUDA_VISIBLE_DEVICES=-1 nohup python pnebo.py with "$2" acq_name="$3" seed="$i" > out/"""$1""_""$2""_""$3""_seed""$i""" &
  done
elif [[ "$1" == "mne" ]]
then
    for i in {1..4}
  do
    echo "Running MNE with ""$2"" acq_name=""$3"" seed=""$i"""
    CUDA_VISIBLE_DEVICES=-1 nohup python mnebo.py with "$2" acq_name="$3" seed="$i" > out/"""$1""_""$2""_""$3""_seed""$i""" &
  done
fi
