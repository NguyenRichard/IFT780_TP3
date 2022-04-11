#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License: Opensource, free to use
Other: Suggestions are welcome
"""

import argparse
import os.path

import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd

from os import mkdir
from os.path import join
from os.path import exists

from CNNTrainTestManager import CNNTrainTestManager, optimizer_setup
from HDF5Dataset import HDF5Dataset
from models.CNN import CNNet
from models.CNNFullNet import FullNet
from models.UNet import UNet
from models.ResNet import ResNet
from transforms import identity
from transforms import crop_and_hflip, rotate_and_blur


def argument_parser():
    """
        A parser to allow user to easily experiment different models along with
        datasets and differents parameters
    """
    parser = argparse.ArgumentParser(usage='\n python3 train.py [model] [dataset] [hyper_parameters]'
                                           '\n python3 train.py --model UNet [hyper_parameters]'
                                           '\n python3 train.py --model UNet --predict',
                                     description="This program allows to train different models of classification on"
                                                 " different datasets. Be aware that when using UNet model there is no"
                                                 " need to provide a dataset since UNet model only train "
                                                 "on acdc dataset.")
    parser.add_argument("exp_name", type=str,
                        help="Name of experiment")
    parser.add_argument('--model', type=str, default="CNNet",
                        choices=["CNNet", "FullNet", "UNet", "UNetDense", "ResNet"])
    parser.add_argument('--dataset_file', type=str,
                        help="Location of the hdf5 file")
    parser.add_argument('--batch_size', type=int, default=20,
                        help='The size of the training batch')
    parser.add_argument('--optimizer', type=str, default="Adam", choices=["Adam", "SGD"],
                        help="The optimizer to use for training the model")
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='The number of epochs')
    parser.add_argument('--validation', type=float, default=0.1,
                        help='Percentage of training data to use for validation')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--dgr', type=int, default=12,
                        help="The growth rate of dense blocks")
    parser.add_argument('--dnl', type=int, default=10,
                        help="The number of layer in dense blocks")
    parser.add_argument('--data_aug', action='store_true',
                        help="Data augmentation")
    parser.add_argument('--data_aug_type', type=str, default="crop_and_hflip", choices=["crop_and_hflip", "rotate_and_blur"],
                        help="The type of data augmentation used")
    parser.add_argument('--load_checkpoint', action='store_true',
                        help="Load saved checkpoint")
    parser.add_argument('--save_checkpoint', action='store_true',
                        help="Save checkpoint")
    parser.add_argument('--checkpoint_filename', type=str, default="checkpoint",
                        help='Name of the checkpoint file')
    parser.add_argument('--predict', action='store_true',
                        help="Load weights to predict the mask of a randomly selected image from the test set")
    parser.add_argument('--bilinear', type=bool, default=False,
                        help="Way of upsampling, bilinear or not")
    return parser.parse_args()


if __name__ == "__main__":
    args = argument_parser()

    path = '' + str(args.model)

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    val_set = args.validation
    learning_rate = args.lr
    data_augment = args.data_aug
    num_layers = args.dnl
    growth_rate = args.dgr
    load_checkpoint = args.load_checkpoint
    save_checkpoint = args.save_checkpoint
    checkpoint_filename = args.checkpoint_filename
    data_augment_type = args.data_aug_type
    bilinear = args.bilinear

    print("Learing rate : ",learning_rate)

    # set hdf5 path according your hdf5 file location
    hdf5_file = args.dataset_file

    # Transform is used to normalize data and more
    if data_augment_type == 'crop_and_hflip':
        data_augment_transform = [
            crop_and_hflip
        ]
    elif data_augment_type == 'rotate_and_blur':
        data_augment_transform = [
            rotate_and_blur
        ]
    else:
        data_augment_transform = {
            identity
        }

    if data_augment:
        print('Using data augmentation')
        transforms = data_augment_transform
    else:
        print("Not using data augmentation")
        transforms = []

    train_set = HDF5Dataset(
        'train', hdf5_file, transforms=transforms)
    test_set = HDF5Dataset(
        'test', hdf5_file, transforms=transforms)
    num_classes = train_set.num_classes
    num_modalities = train_set.num_modalities

    if args.optimizer == 'SGD':
        optimizer_factory = optimizer_setup(torch.optim.SGD, lr=learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer_factory = optimizer_setup(optim.Adam, lr=learning_rate)

    if args.model == 'CNNet':
        model = CNNet(num_classes=num_classes, in_channels=num_modalities)
    elif args.model == 'FullNet':
        model = FullNet(num_classes=num_classes, in_channels=num_modalities, num_layers=num_layers, growth_rate=growth_rate)
    elif args.model == 'UNet':
        model = UNet(num_classes=num_classes, num_channels=num_modalities, bilinear=bilinear)
    elif args.model == 'UNetDense':
        model = UNet(num_classes=num_classes, num_channels=num_modalities, bilinear=bilinear, dense = True)
    elif args.model == 'ResNet':
        model = ResNet.ResNet50(num_classes, num_modalities) 


    model_trainer = CNNTrainTestManager(model=model,
                                        trainset=train_set,
                                        testset=test_set,
                                        batch_size=batch_size,
                                        loss_fn=nn.CrossEntropyLoss(),
                                        optimizer_factory=optimizer_factory,
                                        validation=val_set,
                                        use_cuda=True,
                                        save_checkpoint_folder_path=args.exp_name,
                                        save_checkpoint_filename=checkpoint_filename,
                                        load_checkpoint=load_checkpoint,
                                        save_checkpoint=save_checkpoint,
                                        data_augmentation=data_augment,
                                        data_augmentation_type=data_augment_type
                                        )

    dice = 0
    if exists(join(path, args.model + '.pt')):
        model.load_weights(join(args.model, args.model + '.pt'))
        dice = model_trainer.evaluate_on_test_set()
        print("predicting the mask of a randomly selected image from test set")
        for i in range(2, 4):
            model_trainer.plot_image_mask_prediction(path , i,learning_rate)
    else:
        if not os.path.exists(args.exp_name):
            mkdir(args.exp_name)
        print("Training {} for {} epochs".format(
            model.__class__.__name__, args.num_epochs))
        model_trainer.train(num_epochs)
        dice = model_trainer.evaluate_on_test_set()
        # save the model's weights for prediction (see help for more details)
        model.save(args.exp_name)

        model_trainer.plot_image_mask_prediction(args.exp_name,2,learning_rate)# 2 is the number write on the fig you save
        model_trainer.plot_metrics(args.exp_name)

    if os.path.isfile("Lr&Dice.csv"):
        df = pd.read_csv("Lr&Dice.csv")
    else :
        df = pd.DataFrame(columns=["Model","Learning Rate","Dice"])

    df2 = {'Model': str(args.model), 'Learning Rate': learning_rate, 'Dice': dice}
    df = df.append(df2, ignore_index = True)
    df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    df.to_csv("Lr&Dice.csv")

