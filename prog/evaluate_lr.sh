#!/bin/bash

source env/bin/activate

case $1 in
 -[h?] | --help)
    cat <<-____HALP
        Usage: ${0##*/} [ --help ]
        Launch a training with several learning rate.
        Args:
          1.name of the model: FullNet, UNet, ResNet
          2.number of epochs
          3.initial learning rate
          4.last learning rate
          5.increment of learning rate
        Exemple: ./evaluate_lr.sh FullNet 1 0.01 0.05 0.005

____HALP
        exit 0;;
esac


model=$1
nb_epochs=$2
initial_lr=$3
end_lr=$4
inc=$5



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

