import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):

    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.bias = bias
        padding = int((kernel_size - 1) / 2)

        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size, padding=padding, bias=bias)
        # Note: hidden/cell states are not intialized in construction
        self.hidden_states, self.cell_state = None, None
        self.h_id = id(self.hidden_states)
        self.c_id = id(self.cell_state)
        self.conv1x1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=1)

    def initialize(self, inputs):

        device = inputs.device  # "cpu" or "cuda"
        batch_size, _, height, width = inputs.size()

        # initialize both hidden and cell states to all zeros
        self.hidden_states = torch.zeros(batch_size,
                                         self.hidden_channels, height, width, device=device)
        self.cell_states = torch.zeros(batch_size,
                                       self.hidden_channels, height, width, device=device)
    def initialize_ori(self, inputs):

        device = inputs.device  # "cpu" or "cuda"
        batch_size, channel, height, width = inputs.size()

        # initialize both hidden and cell states to first frames
        self.hidden_states = self.conv1x1(inputs)
        self.cell_states = self.conv1x1(inputs)


    def forward(self, inputs, first_step=False):
        """
        input: 4-th order tensor of size [batch_size, input_channels, heigh, width]
        """

        if first_step: self.initialize(inputs)
        # if first_step: self.initialize_ori(inputs)
        # h_cur, c_cur = cur_state


        combined_conv = self.conv(torch.cat([inputs, self.hidden_states], dim=1))
        # combined_conv = self.groupnorm(combined_conv)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)

        i = torch.sigmoid(cc_i)  # 输入门
        f = torch.sigmoid(cc_f)  # 遗忘门
        o = torch.sigmoid(cc_o)  # 输出门
        g = torch.tanh(cc_g)  # 输入值

        self.cell_states = f * self.cell_states + i * g
        self.hidden_states = o * torch.tanh(self.cell_states)

        return self.hidden_states