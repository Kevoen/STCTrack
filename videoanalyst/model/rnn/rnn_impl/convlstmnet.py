import torch
import torch.nn as nn
from .convlstmcell import ConvLSTMCell
from torch.autograd import Variable

from videoanalyst.model.rnn.rnn_base import TRACK_RNN
from videoanalyst.model.module_base import ModuleBase

class ConvLSTMNet(nn.Module):
    def __init__(self, input_channels,  # the number of channels for input features
                 layers_per_block,  # Number of Conv-LSTM layers in each block
                 hidden_channels,  # the number of output , list of ints, note: the length of hidden_channels (lay)
                 skip_stride=None,
                 kernel_size=3,
                 bias=True):

        super(ConvLSTMNet, self).__init__()

        ## Hyperparameters
        self.layers_per_block = layers_per_block
        self.hidden_channels = hidden_channels

        self.num_blocks = len(layers_per_block)
        assert self.num_blocks == len(hidden_channels), "Invalid number of blocks."

        self.skip_stride = (self.num_blocks + 1) if skip_stride is None else skip_stride

        Cell = lambda in_channels, out_channels: ConvLSTMCell(
            input_channels=in_channels, hidden_channels=out_channels,
            kernel_size=kernel_size, bias=bias)

        ## Construction of convolutional LSTM network
        # stack the convolutional-LSTM layers with skip connections
        self.layers = nn.ModuleDict()
        for b in range(self.num_blocks):
            for l in range(layers_per_block[b]):
                # number of input channels to the current layer
                if l > 0:
                    channels = hidden_channels[b]
                elif b == 0:  # if l == 0 and b == 0:
                    channels = input_channels
                else:  # if l == 0 and b > 0:
                    channels = hidden_channels[b - 1]
                    if b > self.skip_stride:
                        channels += hidden_channels[b - 1 - self.skip_stride]

                lid = "b{}l{}".format(b, l)  # layer ID
                self.layers[lid] = Cell(channels, hidden_channels[b])
        # number of input channels to the last layer (output layer)
        channels = hidden_channels[-1]
        if self.num_blocks >= self.skip_stride:
            channels += hidden_channels[-1 - self.skip_stride]

        self.layers["output"] = nn.Conv2d(channels, input_channels,
                                          kernel_size=1, padding=0, bias=True)

    def forward(self, inputs, input_frames, future_frames, output_frames,
                teacher_forcing=False, scheduled_sampling_ratio=0):
        # input: b, t, c, h, w
        # compute the teacher forcing mask
        if teacher_forcing and scheduled_sampling_ratio > 1e-6:
            # generate the teacher_forcing mask (4-th order)
            teacher_forcing_mask = torch.bernoulli(scheduled_sampling_ratio *
                                                   torch.ones(inputs.size(0), future_frames - 1, 1, 1, 1,
                                                              device=inputs.device))
        else:  # if not teacher_forcing or scheduled_sampling_ratio < 1e-6:
            teacher_forcing = False

        # the number of time steps in the computational graph
        total_steps = input_frames + future_frames - 1
        outputs = [None] * total_steps

        for t in range(total_steps):
            # input_: 4-th order tensor of size [batch_size, input_channels, height, width]
            if t < input_frames:
                input_ = inputs[:, t]
            elif not teacher_forcing:
                input_ = outputs[t - 1]
            else:  # if t >= input_frames and teacher_forcing:
                mask = teacher_forcing_mask[:, t - input_frames]
                input_ = inputs[:, t] * mask + outputs[t - 1] * (1 - mask)

            first_step = (t == 0)

            queue = []  # previous outputs for skip connection
            for b in range(self.num_blocks):
                for l in range(self.layers_per_block[b]):
                    lid = "b{}l{}".format(b, l)  # layer ID
                    input_ = self.layers[lid](input_, first_step=first_step)

                queue.append(input_)
                if b >= self.skip_stride:
                    input_ = torch.cat([input_, queue.pop(0)], dim=1)  # concat over the channels

            # map the hidden states to predictive frames (with optional sigmoid function)
            outputs[t] = self.layers["output"](input_)

        # return the last output_frames of the outputs
        outputs = outputs[-output_frames:]

        # 5-th order tensor of size [batch_size, output_frames, channels, height, width]
        return torch.stack([outputs[t] for t in range(output_frames)], dim=1)


@TRACK_RNN.register
class convlstm(ModuleBase):
    default_hyper_params = dict(
        pretrain_model_path="",
        input_channels =  256,
        layers_per_block = (2, 3, 3, 2),
        hidden_channels = (128, 64, 64, 128),
        skip_stride = 2,
        kernel_size = 3,
        bias = True,
    )

    def __init__(self):
        super(convlstm, self).__init__()

    def forward(self, inputs, input_frames, future_frames, output_frames):
        return self.convlstm(inputs, input_frames, future_frames, output_frames)

    def update_params(self):
        super().update_params()
        input_channels = self._hyper_params['input_channels']
        layers_per_block = self._hyper_params['layers_per_block']
        hidden_channels = self._hyper_params['hidden_channels']
        skip_stride = self._hyper_params['skip_stride']
        kernel_size = self._hyper_params['kernel_size']
        bias = self._hyper_params['bias']
        self.convlstm = ConvLSTMNet(input_channels,
                                    layers_per_block,
                                    hidden_channels,
                                    skip_stride,
                                    kernel_size,
                                    bias)
        print(self.convlstm)

if __name__ == '__main__':
    from thop import profile
    import torch
    from ptflops import get_model_complexity_info
    model = ConvLSTMNet()
    x = torch.randn(1, 3, 224, 224)
    fb = torch.randn(1, 1, 224, 224)
    flops, params = profile(model, inputs=(x, fb, ))
    # print("flops:", flops / 1e9,"G params：“ ,params / 1e6)  # flops单位G，para单位M
    print("FLOPS:{} G, Params: {} M".format(flops / 1e9, params / 1e6))