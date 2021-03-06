# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import warnings
import torch
import numpy as np
from DataManager import DataManager
from typing import Callable, Type
from tqdm import tqdm
from os.path import join
from utils import mean_dice, convert_mask_to_rgb_image
import matplotlib.pyplot as plt
import os.path

class CNNTrainTestManager(object):
    """
    Class used the train and test the given model in the parameters
    """

    def __init__(self, model,
                 trainset: torch.utils.data.Dataset,
                 testset: torch.utils.data.Dataset,
                 loss_fn: torch.nn.Module,
                 optimizer_factory: (Callable[[torch.nn.Module],
                                              torch.optim.Optimizer]),
                 batch_size=1,
                 validation=None,
                 use_cuda=False,
                 save_checkpoint_folder_path="",
                 save_checkpoint_filename="checkpoint",
                 save_checkpoint=False,
                 load_checkpoint=False,
                 data_augmentation=False,
                 data_augmentation_type="crop_and_hflip"
                 ):
                 
        """
        Args:
            model: model to train
            trainset: dataset used to train the model
            testset: dataset used to test the model
            loss_fn: the loss function used
            optimizer_factory: A callable to create the optimizer. see
                optimizer function below for more details
            validation: wether to use custom validation data or let the one by
                default
            use_cuda: to Use the gpu to train the model
            save_checkpoint_folder_path(string): path to the folder to save the checkpoint
            save_checkpoint_filename(name): name of the checkpoint file
            save_checkpoint(bool): activate checkpoint saving
            load_checkpoint(bool): activate checkpoint loading
            data_augmentation: whether
            data_augmentation_type: type of data_augmentation
        """

        device_name = 'cuda:0' if use_cuda else 'cpu'
        if use_cuda and not torch.cuda.is_available():
            warnings.warn("CUDA is not available. Suppress this warning by"
                          "passing use_cuda=False to {}()."
                          .format(self.__class__.__name__), RuntimeWarning)
            device_name = 'cpu'

        self.device = torch.device(device_name)
        if validation is not None:
            self.data_manager = DataManager(
                trainset, testset, batch_size=batch_size,
                validation=validation, num_workers=8)
        else:
            self.data_manager = DataManager(
                trainset, testset, batch_size=batch_size, num_workers=8)
        self.loss_fn = loss_fn
        self.model = model
        self.optimizer = optimizer_factory(self.model)
        self.model = self.model.to(self.device)
        self.use_cuda = use_cuda
        self.metric_values = {}
        self.checkpoint = {}
        self.checkpoint_folder_path = save_checkpoint_folder_path
        self.checkpoint_filename = save_checkpoint_filename
        self.save_checkpoint = save_checkpoint
        self.load_checkpoint = load_checkpoint
        self.start_epoch = 0
        self.data_augmentation = data_augmentation
        self.data_augmentation_type = data_augmentation_type

    def save(self, epoch):
        self.checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss': self.loss_fn,
            'metric_values': self.metric_values
        }
        path = self.checkpoint_folder_path + "/" + self.checkpoint_filename + ".pt"
        torch.save(self.checkpoint, path)
        print("Checkpoint epoch", epoch+1, " saved.")

    def load(self):
        path = self.checkpoint_folder_path + "/" + self.checkpoint_filename + ".pt"
        if os.path.exists(path):
            self.checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            self.metric_values = self.checkpoint['metric_values']
            self.loss_fn = self.checkpoint['loss']
            self.start_epoch = self.checkpoint['epoch'] + 1

    def augment_data(self, images, labels):
        if self.data_augmentation_type == "crop_and_hflip":
            return self.data_manager.augment_data_crop_and_hflip(images, labels, (244, 244), 2)


    def train(self, num_epochs):
        """
        Train the model for num_epochs times
        Args:
            num_epochs: number times to train the model
        """
        # Initialize metrics container
        self.metric_values['train_loss'] = []
        self.metric_values['train_acc'] = []
        self.metric_values['val_loss'] = []
        self.metric_values['val_acc'] = []

        # Load checkpoint
        if self.load_checkpoint:
            self.load()
        
        # Create pytorch's train data_loader
        train_loader = self.data_manager.get_train_set()

        # train num_epochs times
        for epoch in range(self.start_epoch, num_epochs):
            print("Epoch: {} of {}".format(epoch + 1, num_epochs))
            train_loss = 0.0

            with tqdm(range(len(train_loader))) as t:
                train_losses = []
                train_accuracies = []
                for i, data in enumerate(train_loader, 0):
                    # transfer tensors to selected device
                    train_inputs, train_labels = \
                        data[0].to(self.device, dtype=torch.float), \
                        data[1].to(self.device, dtype=torch.long)
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward pass
                    train_outputs = self.model(train_inputs)
                    # computes loss using loss function loss_fn
                    loss = self.loss_fn(train_outputs, train_labels)

                    # Use autograd to compute the backward pass.
                    loss.backward()

                    # updates the weights using gradient descent
                    """
                    Way it could be done manually

                    with torch.no_grad():
                        for param in self.model.parameters():
                            param -= learning_rate * param.grad
                    """
                    self.optimizer.step()

                    # Save losses for plotting purposes
                    train_losses.append(loss.item())
                    train_accuracies.append(
                        self.accuracy(train_outputs, train_labels))

                    # print metrics along progress bar
                    train_loss += loss.item()
                    t.set_postfix(loss='{:05.3f}'.format(train_loss / (i + 1)))
                    t.update()
            # evaluate the model on validation data after each epoch
            self.metric_values['train_loss'].append(np.mean(train_losses))
            self.metric_values['train_acc'].append(np.mean(train_accuracies))
            self.evaluate_on_validation_set()

            #Save checkpoint
            if self.save_checkpoint:
                self.save(epoch)

        print("Finished training.")

    def evaluate_on_validation_set(self):
        """
        function that evaluate the model on the validation set every epoch
        """
        # switch to eval mode so that layers like batchnorm's layers nor
        # dropout's layers works in eval mode instead of training mode
        self.model.eval()

        # Get validation data
        val_loader = self.data_manager.get_validation_set()
        validation_loss = 0.0
        validation_losses = []
        validation_accuracies = []

        with torch.no_grad():
            for j, val_data in enumerate(val_loader, 0):
                # transfer tensors to the selected device
                val_inputs, val_labels = val_data[0].to(
                    self.device, dtype=torch.float), \
                    val_data[1].to(self.device, dtype=torch.long)

                # forward pass
                val_outputs = self.model(val_inputs)

                # compute loss function
                loss = self.loss_fn(val_outputs, val_labels)
                validation_losses.append(loss.item())
                validation_accuracies.append(
                    self.accuracy(val_outputs, val_labels))
                validation_loss += loss.item()

        self.metric_values['val_loss'].append(np.mean(validation_losses))
        self.metric_values['val_acc'].append(np.mean(validation_accuracies))

        # displays metrics
        print('Validation loss %.3f' % (validation_loss / len(val_loader)))

        # switch back to train mode
        self.model.train()

    def accuracy(self, outputs, labels):
        """
        Computes the accuracy of the model
        Args:
            outputs: outputs predicted by the model
            labels: real outputs of the data
        Returns:
            Accuracy of the model
        """
        # compute the mean of the 3 classes's dice score
        return mean_dice(outputs, labels).item()

    def evaluate_on_test_set(self):
      """
      Evaluate the model on the test set
      :returns;
          Accuracy of the model on the test set
      """
      test_loader = self.data_manager.get_test_set()
      accuracies = 0
      with torch.no_grad():
          for data in test_loader:
              test_inputs, test_labels = data[0].to(
                  self.device, dtype=torch.float), data[1].to(
                      self.device, dtype=torch.long)
              test_outputs = self.model(test_inputs)
              assert torch.where(test_labels)
              accuracies += self.accuracy(test_outputs, test_labels)
      dice = accuracies / len(test_loader)
      print("Dice on the test set: {:05.3f} %".format(
          dice))
      return dice 

    def plot_metrics(self, path):
        """
        Function that plots train and validation losses and accuracies after
        training phase
        """
        epochs = range(1, len(self.metric_values['train_loss']) + 1)

        f = plt.figure(figsize=(10, 5))
        ax1 = f.add_subplot(121)
        ax2 = f.add_subplot(122)

        # loss plot
        ax1.plot(
            epochs, self.metric_values['train_loss'],
            '-o', label='Training loss')
        ax1.plot(
            epochs, self.metric_values['val_loss'],
            '-o', label='Validation loss')
        ax1.set_title('Training and validation loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # accuracy plot
        ax2.plot(
            epochs, self.metric_values['train_acc'], '-o',
            label='Training accuracy')
        ax2.plot(
            epochs, self.metric_values['val_acc'], '-o',
            label='Validation accuracy')
        ax2.set_title('Training and validation accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('accuracy')
        ax2.legend()

        f.savefig(join(path, 'fig1.png'))
        #plt.show()

    def plot_image_mask_prediction(self, path,number,lr):
        """
        Function that plots an image its corresponding ground truth and
        the predicted mask
        """
        # pick randomly an image and its corresponding gt into test_set
        img, gt = self.data_manager.get_random_sample_from_test_set()

        assert torch.any(gt)

        # convert the ground truth to a rgb image
        gt = convert_mask_to_rgb_image(gt)
        # use the model to predict the segmented image
        # Since the model expect a 4D we need to add the batch dim in order
        # to get the 4D
        img = np.expand_dims(img, axis=0)
        # convert image to Tensor
        img = torch.from_numpy(img)
        img = img.to(self.device, dtype=torch.float)
        prediction = self.model(img)
        # delete the batch dimension
        prediction = np.squeeze(prediction)
        # convert prediction to numpy array
        # take into account if model is trained on cpu or gpu
        if self.use_cuda and torch.cuda.is_available():
            prediction = prediction.detach().cpu().numpy()
        else:
            prediction = prediction.detach().numpy()
        # from one_hot vector to categorical
        prediction = np.argmax(prediction, axis=0)
        # convert the predicted mask to rgb image
        prediction = convert_mask_to_rgb_image(prediction)
        # remove the batch dim and the channel dim of img
        img = np.squeeze(np.squeeze(img))
        if len(img.shape) == 3:
            img = img[0, ...]
        # convert img to a numpy array
        if self.use_cuda and torch.cuda.is_available():
            img = img.cpu().numpy()
        else:
            img = img.numpy()

        # plot the image
        f = plt.figure(figsize=(10, 10))
        ax1 = f.add_subplot(221)
        ax1.imshow(img, cmap='gray')
        ax1.set_title('image')
        ax1.axis('off')

        # plot the ground truth
        ax2 = f.add_subplot(222)
        ax2.imshow(gt.astype('uint8'), cmap='gray')
        ax2.axis('off')
        ax2.set_title('ground truth')

        # plot the image
        ax3 = f.add_subplot(223)
        ax3.imshow(img, cmap='gray')
        ax3.set_title('image')
        ax3.axis('off')

        # plot the predicted mask
        ax4 = f.add_subplot(224)
        ax4.imshow(prediction.astype('uint8'), cmap='gray')
        ax4.set_title('predicted segmentation')
        ax4.axis('off')

        # Save as a png image
        print("Figure saved at : ", join(path, 'fig'+str(number)+'_'+str(lr)+'.png'))
        f.savefig(join(path, 'fig'+str(number)+'_'+str(lr)+'.png'))
        # show image
        #plt.show()


def optimizer_setup(
    optimizer_class: Type[torch.optim.Optimizer],
    **hyperparameters
) -> \
        Callable[[torch.nn.Module], torch.optim.Optimizer]:
    """
    Creates a factory method that can instanciate optimizer_class with the
    given hyperparameters.

    Why this? torch.optim.Optimizer takes the model's parameters as an
    argument. Thus we cannot pass an Optimizer to the CNNBase
    constructor.

    Args:
        optimizer_class: optimizer used to train the model
        **hyperparameters: hyperparameters for the model
        Returns:
            function to setup the optimizer
    """

    def f(model):
        return optimizer_class(model.parameters(), **hyperparameters)

    return f
