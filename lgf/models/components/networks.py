import torch
import torch.nn as nn

from .helpers import ScaledTanh2dModule


class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = self._get_conv3x3(num_channels)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = self._get_conv3x3(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        out = self.bn1(inputs)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = out + inputs

        return out

    def _get_conv3x3(self, num_channels):
        return nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )


def get_resnet(
        num_input_channels,
        hidden_channels,
        num_output_channels
):
    num_hidden_channels = hidden_channels[0] if hidden_channels else num_output_channels

    layers = [
        nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=num_hidden_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
    ]

    for num_hidden_channels in hidden_channels:
        layers.append(ResidualBlock(num_hidden_channels))

    layers += [
        nn.BatchNorm2d(num_hidden_channels),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=num_hidden_channels,
            out_channels=num_output_channels,
            kernel_size=1
        )
    ]

    return ScaledTanh2dModule(
        module=nn.Sequential(*layers),
        num_channels=num_output_channels
    )


def get_mlp(num_inputs, hidden_units, num_outputs, activation, log_softmax_outputs=False):
    layers = []
    prev_num_hidden_units = num_inputs
    for num_hidden_units in hidden_units:
        layers.append(nn.Linear(prev_num_hidden_units, num_hidden_units))
        layers.append(activation())
        prev_num_hidden_units = num_hidden_units
    layers.append(nn.Linear(prev_num_hidden_units, num_outputs))

    if log_softmax_outputs:
        layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)


def get_glow_cnn(num_input_channels, num_hidden_channels, num_output_channels):
    conv1 = nn.Conv2d(
        in_channels=num_input_channels,
        out_channels=num_hidden_channels,
        kernel_size=3,
        padding=1,
        bias=False
    )

    conv2 = nn.Conv2d(
        in_channels=num_hidden_channels,
        out_channels=num_hidden_channels,
        kernel_size=1,
        bias=False
    )

    conv3 = nn.Conv2d(
        in_channels=num_hidden_channels,
        out_channels=num_output_channels,
        kernel_size=3,
        padding=1
    )
    conv3.weight.data.zero_()
    conv3.bias.data.zero_()

    batchnorm1 = nn.BatchNorm2d(num_hidden_channels)
    batchnorm2 = nn.BatchNorm2d(num_hidden_channels)

    return nn.Sequential(
        conv1,
        batchnorm1,
        nn.ReLU(),
        conv2,
        batchnorm2,
        nn.ReLU(),
        conv3
    )
