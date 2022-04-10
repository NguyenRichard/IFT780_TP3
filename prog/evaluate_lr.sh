#!/bin/bash

source env/bin/activate

initial_lr=0.01
end_lr=0.05
inc=0.005

nb_epochs=1

default_directory="acdc_train"

for i in $(LC_ALL=en_US.UTF-8 seq $initial_lr $inc $end_lr)
do
  save_directory=$default_directory"/"${i//[.]/_}
  python train.py $save_directory --data_aug --model=FullNet --dgr=4 --dnl=4 --lr=$i --load_checkpoint --save_checkpoint --dataset='./02Heart.hdf5' --num-epochs=$nb_epochs --batch_size=2

done

