#!/bin/bash

source env/bin/activate

model=$1
initial_lr=$2
end_lr=$3
inc=$4

nb_epochs=1

default_directory="acdc_train"

for i in $(LC_ALL=en_US.UTF-8 seq $initial_lr $inc $end_lr)
do
  save_directory=$default_directory"/"${i//[.]/_}
  if [ "$model" = "FullNet" ]; then
    echo "Start training with learning rate $i."
    python train.py $save_directory --data_aug --model=FullNet --dgr=4 --dnl=4 --lr=$i --load_checkpoint --save_checkpoint --dataset='./02Heart.hdf5' --num-epochs=$nb_epochs --batch_size=2
  elif [ "$model" = "UNet" ]; then
    echo "TODO add UNet command"
  elif [ "$model" = "ResNet" ]; then
    echo "TODO add ResNet command"
  fi

done

