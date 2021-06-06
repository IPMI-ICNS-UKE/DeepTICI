import math
import os
from typing import Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torch.cuda.amp import autocast

from helper import ModelMode, OutputMode


class SwishImplementation(torch.autograd.Function):
    """legacy method used before torch.nn.SiLU was available"""

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(nn.Module):
    """pytorch implementation of Swish activation function"""

    def forward(self, x):
        return SwishImplementation.apply(x)


class EfficientTwoArmEncoder(nn.Module):
    """Implements a two-arm-encoder network based on an efficient-net b0 backbone. This networks fuses each frame of two
     views to one latent representation by combining them before the last convolutional layer."""

    def __init__(self, num_classes: int = 7, in_channels: int = 3, feature_size: int = 1280):
        """args:
            num_classes: Number of classes if this network is used for classification.
            in_channels: Number of channels in the input image
            feature_size: size of the latent space"""
        super().__init__()
        # Efficient-net backbone
        self.feature_extractor = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes,
                                                              in_channels=in_channels)
        self.swish = Swish()

        # combines latent representation of the two views
        self.combine_layer = nn.Sequential(
            nn.Conv2d(320 * 2, feature_size, 1, bias=False),
            nn.BatchNorm2d(feature_size, momentum=0.01, eps=1e-3),
            self.swish
        )

        # classification layer
        self.final_layer = nn.Sequential(
            nn.Linear(feature_size, num_classes),
            Swish(),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """args:
            x: Tensor of size batch_size x 2 (num_views) x frames x channels x height x width
            returns: latent representation of x batch_size x frames x feature_size"""

        x_shape = x.shape
        # flatten image to batch_size*views*frames x channels x height x width
        x = x.contiguous().view((-1, *x_shape[-3:]))

        # encode all frames and views simultaneously
        x = self.extract_features(x)

        # unflatten to dimension: batch_size*frames x channels*views x height x width
        x = x.view((-1, x.shape[-3] * 2, *x.shape[-2:]))
        x = self.combine_layer(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(dim=2).squeeze(dim=2)

        # reshape to batch_size x frames x feature_size
        x = x.view((*x_shape[:2], x.shape[-1]))
        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """applies all layers of efficient-net b0 until the last convolution
        args:
            x: input tensor shape: n x channels x height x width"""
        x = self.swish(self.feature_extractor._bn0(self.feature_extractor._conv_stem(x)))
        for idx, block in enumerate(self.feature_extractor._blocks):
            drop_connect_rate = self.feature_extractor._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.feature_extractor._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
        return x


class FastGRU(nn.Module):
    def __init__(self, feature_size: int, output_size: int = None):
        """intializes weights of the GRU with three gates: Input, forget and hidden.
        args:
            feature_size: Size of the input in the last dimension
            output_size: size of the cell state/ output. if not given it is equal to feature_size"""
        super().__init__()
        self.input_size = feature_size
        if not output_size:
            self.hidden_size = self.input_size
        else:
            self.hidden_size = output_size
        self.W = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size * 3))
        self.U = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size * 3))
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size * 3))
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """implements GRU forward pass with three gates. Input, forget and output. The input gate controls the relevance
         of new timesteps, the forget gate the relevance of the cell state in regard to the new input and the output
         gate controls the new cellstate.
         args:
            x: input tensor of shape batch x timesteps x feature_size"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        dropout_prob = 0.1
        h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device),
                    torch.zeros(bs, self.hidden_size).to(x.device))

        x = F.dropout(input=x, p=dropout_prob, training=self.training)

        for t in range(seq_sz):
            x_t = x[:, t, :]

            gates = x_t @ self.U + c_t @ self.W + self.bias

            z_t, r_t = (
                # input
                torch.sigmoid(gates[:, :self.hidden_size]),
                # forget
                torch.sigmoid(gates[:, self.hidden_size:self.hidden_size * 2])
            )

            x_t = torch.tanh(r_t * x_t @ self.U + c_t @ self.W + self.bias)[:,
                  self.hidden_size * 2:self.hidden_size * 3]

            c_t = (1 - z_t) * c_t + x_t

            hidden_seq.append(c_t.unsqueeze(0))

            c_t = F.dropout(input=c_t, p=dropout_prob, training=self.training)

        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        return hidden_seq


class TICIModelHandler(nn.Module):
    def __init__(self,
                 num_classes: int,
                 feature_size: int,
                 pretrained: Union[None, Union[str, os.PathLike], List[Union[str, os.PathLike]]] = None,
                 in_channels: int = 3,
                 output_size: int = None
                 ):
        """Wrapper around Encoder+GRU+Classifier structure. Serves model loading and selecting the right timesteps for
        training and inference.
        args:
            num_classes: number of classes in the output
            feature_size: output size of the encoder network
            pretrained: path(s) to pickeld weights, if multiple paths are given an ensamble is applied
            in_channels: expected number of channels in the input image
            output_size: size of the cell state for the GRU. If not given equal to feature_size"""
        super().__init__()
        self.num_classes = num_classes
        self.network = TICITemporalNetwork(in_channels=in_channels, feature_size=feature_size, output_size=output_size,
                                           num_classes=num_classes)
        self.pretrained = False
        if pretrained:
            if isinstance(pretrained, list):
                self.pretrained = pretrained
            else:
                # if multiple paths ensemble over all given weights
                self.pretrained = True
                self.load_model(pretrained)

    def load_model(self, path):
        """load state dict of the network.
        args:
            path: path to weights"""
        state_dict = torch.load(path)
        while any(['module.' in k for k in state_dict.keys()]):
            # remove eventual pytorch wrapper
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.load_state_dict(state_dict, strict=False)

    @autocast()
    def forward(self, x: torch.Tensor, series_length: torch.Tensor = None, model_mode: ModelMode = ModelMode.inference,
                output_mode: OutputMode = OutputMode.last_frame) -> torch.tensor:
        if x.dim() == 5:
            x = x.unsqueeze(dim=0)
        assert x.dim() == 6, 'Input must be 4 or 5 dimensional ((batch) x time x view x channels x height x width'
        if series_length:
            if series_length.dim() == 0:
                series_length = series_length.unsqueeze(dim=0)

        series_length = series_length.to(x.device)
        assert isinstance(model_mode, ModelMode), 'forward pass mode is not implemented (yet)'

        # inference
        if model_mode == ModelMode.inference:
            with torch.no_grad():
                # ensemble
                if isinstance(self.pretrained, list):
                    for path in self.pretrained:
                        self.load_model(path)
                        self.eval()
                        try:
                            predictions += self.network(x)
                        except NameError:
                            predictions = self.network(x)
                    predictions /= len(self.pretrained)
                else:
                    self.eval()
                    predictions = self.network(x)
        # training
        elif model_mode == ModelMode.train:
            self.train()
            predictions = self.network(x)
        else:
            raise NotImplementedError('forward pass mode is not implemented (yet)')

        assert isinstance(output_mode, OutputMode), 'output mode is not implemented (yet)'
        if output_mode == OutputMode.last_frame:
            predictions = self._get_last_frame_in_batch(predictions, series_length)
        elif output_mode == OutputMode.all_frames:
            pass

        return predictions

    @staticmethod
    def _get_last_frame_in_batch(predictions: torch.Tensor, series_length: torch.Tensor = None) -> torch.Tensor:
        """iterates through batch to fetch the last frame of each input
        args:
            predictions: tensor of shape batch x time x num_classes
            series_length: tensor containing the series lengths of batch elements. If not specified the last frame is
            considered"""
        if not series_length:
            pred = predictions[:, -1]
        else:
            pred = torch.zeros((predictions.shape[0], predictions.shape[-1]), device=predictions.device)
            for i, _series_length in enumerate(series_length):
                pred[i] = predictions[i, _series_length - 1]
        return pred


class TICITemporalNetwork(nn.Module):
    def __init__(self,
                 in_channels: int,
                 output_size: int,
                 num_classes: int,
                 feature_size: int) -> torch.Tensor:
        """Wraps around encoder, GRU and classifier.
        args:
            in_channels: number of channels in input
            output_size: size of the GRUs cell state
            num_classes: number of classes in the dataset
            feature_size: number of features from the encoder"""
        super().__init__()
        self.encoder = EfficientTwoArmEncoder(feature_size=feature_size, in_channels=in_channels,
                                              num_classes=num_classes)
        self.gru = FastGRU(feature_size=feature_size, output_size=output_size)
        self.classifier = nn.Sequential(nn.Dropout(),
                                        nn.Linear(feature_size, num_classes),
                                        Swish(),
                                        nn.Softmax(dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward pass through all model parts:
        args:
            x: input tensor of shape batch x time x views x channels x height x width"""
        x = self.encoder(x, mode='encoder')
        x = self.gru(x)
        x = self.classifier(x)
        return x
