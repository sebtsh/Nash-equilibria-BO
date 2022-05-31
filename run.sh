echo "exp_type: $1"
echo "utility_name: $2";
echo "acq_name: $3";
echo "start seed: $4";
echo "end seed: $5";

mkdir out

if [[ "$1" == "pne" ]]
then
  for ((i=$4; i<=$5; i++))
  do
    echo "Running PNE with $2 acq_name=$3 seed=$i"
    CUDA_VISIBLE_DEVICES=-1 nohup python pnebo.py with "$2" acq_name="$3" seed="$i" > out/"$1_$2_$3_seed$i.txt" &
  done
elif [[ "$1" == "mne" ]]
then
    for ((i=$4; i<=$5; i++))
  do
    echo "Running MNE with $2 acq_name=$3 seed=$i"
    CUDA_VISIBLE_DEVICES=-1 nohup python mnebo.py with "$2" acq_name="$3" seed="$i" > out/"$1_$2_$3_seed$i.txt" &
  done
fi
