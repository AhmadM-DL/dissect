# -*- coding: utf-8 -*-
"""
Created on Tuesday April 14 2020
@author: Ahmad Mustapha (amm90@mail.aub.edu)
"""
import torch
import math
import time
import os
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from torch.utils.tensorboard import SummaryWriter


# -*- coding: utf-8 -*-
import torch

class SobelFilter(torch.nn.Module):
    def __init__(self):
        """
        In this constructor we initialize the Sobel Filter Layer which is composed 
        from two Conv2d layers. The first transforms RGB input to grayscale.
        The second apply sobel filter on grayscale input.
        """
        super(SobelFilter, self).__init__()

        self.grayscale = torch.nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
        self.grayscale.weight.data.fill_(1.0 / 3.0)
        self.grayscale.bias.data.zero_()

        self.sobel_filter = torch.nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
        self.sobel_filter.weight.data[0, 0].copy_(
            torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        )
        self.sobel_filter.weight.data[1, 0].copy_(
            torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        )
        self.sobel_filter.bias.data.zero_()

        for p in self.grayscale.parameters():
            p.requires_grad = False

        for p in self.sobel_filter.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.grayscale(x)
        x = self.sobel_filter(x)
        return x


class DeepClusteringNet(torch.nn.Module):

    def __init__(self, input_size, features, classifier, top_layer, device, with_sobel=False, concat_sobel=False):
        super().__init__()
        self.sobel = SobelFilter() if with_sobel else None
        self.features = features
        self.classifier = classifier
        self.top_layer = top_layer
        self.input_size = input_size
        self.concat_sobel = concat_sobel


        self._initialize_weights()
        self.device = device
        self.to(self.device)

        return

    def forward(self, x):
        if self.sobel:
            if self.concat_sobel:
                aug = self.sobel(x)
                x = torch.cat( (x, aug), 1)
            else:
                x = self.sobel(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if self.top_layer:
            x = torch.nn.functional.relu(x)
            x = self.top_layer(x)
        return x
    
    def extract_features(self, x, target_layer, flatten=True):
        if self.sobel:
            x = self.sobel(x)
        
        for module_name, module in self.features.named_children():
            x = module(x)
            if module_name == target_layer:
                break
        if flatten:
            x = x.view(x.size(0), -1)
            
        return x

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def output_size(self, single_input_size):
        """
        A method that computes model output size by feeding forward
        a dummy single input. This method doesn't consider the batch size
        :param single_input_size: tuple
            a tuple that includes the input size in the form of (#Channels, Width, Height)
        :return: tuple
            a tuple that includes the output size
        """
        x = torch.rand(size=(1, *single_input_size), device=self.device)
        x = self.forward(x)
        return tuple(x.size()[1:])

    def add_top_layer(self, output_size):
        # get model output size
        model_output_size = self.output_size(self.input_size)[0]
        linear_layer = torch.nn.Linear(model_output_size, output_size)
        linear_layer.weight.data.normal_(0, 0.01)
        linear_layer.bias.data.zero_()
        self.top_layer = torch.nn.Sequential(torch.nn.ReLU(),linear_layer)
        self.top_layer.to(self.device)

    def remove_top_layer(self):
        self.top_layer == None

    def freeze_features(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_features(self):
        for param in self.features.parameters():
            param.requires_grad = True

    def freeze_classifier(self):
        for param in self.classifier.parameters():
            param.requires_grad = False

    def unfreeze_classifier(self):
        for param in self.classifier.parameters():
            param.requires_grad = True

    def deep_cluster_train(self, dataloader, epoch, optimizer: torch.optim.Optimizer, loss_fn, verbose=False,
                           writer: SummaryWriter = None):

        if verbose:
            print('Training Model')

        self.train()
        end = time.time()

        for i, (input_, target) in enumerate(dataloader):

            input_ = input_.to(self.device)
            target = target.to(self.device)
            output = self(input_)

            loss = loss_fn(output, target)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if writer:

                writer.add_scalar("training_loss",
                                  scalar_value=loss.item(),
                                  global_step=epoch * len(dataloader) + i)

            if verbose and len(dataloader) >= 10 and (i % (len(dataloader)//10)) == 0:
                print('{0} / {1}\tTime: {2:.3f}'.format(i,
                                                        len(dataloader), time.time() - end))

            end = time.time()
    
    def deep_cluster_train_with_weights(self, dataloader,
                           epoch, optimizer: torch.optim.Optimizer,
                           loss_fn, 
                           instance_wise_weights:torch.tensor,
                           verbose=False,
                           writer: SummaryWriter = None,
                           writer_tag= None
                           ):

        if verbose:
            print('Training Model')

        self.train()
        end = time.time()

        if not writer_tag:
            writer_tag= ""
        else:
            writer_tag= "/"+writer_tag

        dataloader.dataset.set_instance_wise_weights(instance_wise_weights)

        for i, (input_, target, instance_wise_weight) in enumerate(dataloader):

            input_ = input_.to(self.device)
            target = target.to(self.device)
            instance_wise_weight = instance_wise_weight.to(self.device)
            instance_wise_weight = torch.as_tensor(instance_wise_weight, dtype=torch.float64)

            output = self(input_)

            loss = loss_fn(output, target)
            loss = torch.as_tensor(loss, dtype=torch.float64)
            if (loss.dim() == 0):
                raise Exception("Error This function expects a loss criterion that doesn't apply reduction\n")
            
            loss = loss * instance_wise_weight
            loss = loss.mean()

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if writer:
                
                writer.add_scalar("training_loss"+writer_tag,
                                  scalar_value=loss.item(),
                                  global_step=epoch * len(dataloader) + i)

            if verbose and len(dataloader) >= 10 and (i % (len(dataloader)//10)) == 0:
                print('{0} / {1}\tTime: {2:.3f}'.format(i,
                                                        len(dataloader), time.time() - end))

            end = time.time()
        
        dataloader.dataset.unset_instance_wise_weights()


    def full_feed_forward(self, dataloader, verbose=False):

        if verbose:
            print('Computing Model Output')

        self.eval()
        end = time.time()

        for i, (input_, _) in enumerate(dataloader):

            batch_size = dataloader.batch_size
            input_ = input_.to(self.device)
            output = self(input_).data.cpu().numpy()

            if i == 0:
                outputs = np.zeros(
                    shape=(len(dataloader.dataset), output.shape[1]), dtype=np.float32)

            if i < len(dataloader) - 1:
                outputs[i * batch_size: (i + 1) *
                        batch_size] = output.astype('float32')
            else:
                # special treatment for final batch
                outputs[i * batch_size:] = output.astype('float32')

            if verbose and len(dataloader) >= 10 and (i % (len(dataloader)//10)) == 0:
                print('{0} / {1}\tTime: {2:.3f}'.format(i,
                                                        len(dataloader), time.time() - end))

            end = time.time()

        return outputs

    def load_model_parameters(self, model_parameters_path, optimizer=None):
        if os.path.isfile(model_parameters_path):
            print("=> loading checkpoint '{}'".format(model_parameters_path))
            checkpoint = torch.load(model_parameters_path)
            start_epoch = checkpoint['epoch']
            self.load_state_dict(checkpoint['state_dict'])
            if optimizer:
                optimizer.load_state_dict(checkpoint['optimizer'])

            print("=> loaded checkpoint '{}' (epoch {})".format(
                model_parameters_path, checkpoint['epoch']))

            return start_epoch
        else:
            print("=> no checkpoint found at '{}'".format(model_parameters_path))
            raise Exception("No checkpoint found at %s" %(model_parameters_path))

    def save_model_parameters(self, model_parameters_path, epoch, optimizer=None):
        model_dict = {'epoch': epoch,
                      'state_dict': self.state_dict()}
        if optimizer:
            model_dict['optimizer'] = optimizer.state_dict()

        torch.save(model_dict, model_parameters_path)

        return