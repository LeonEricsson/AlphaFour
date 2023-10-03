import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlockLN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlockLN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = nn.LayerNorm([out_channels, 6, 7])
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.norm2 = nn.LayerNorm([out_channels, 6, 7])

    def forward(self, x):
        residual = x
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class C4NetLN(nn.Module):
    def __init__(self, num_blocks=3):
        super(C4NetLN, self).__init__()
        self.conv_initial = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.norm_initial = nn.LayerNorm([128, 6, 7])

        # Create Residual Blocks with LayerNorm
        self.res_blocks = nn.ModuleList(
            [ResidualBlockLN(128, 128) for _ in range(num_blocks)]
        )

        self.fc1 = nn.Linear(128 * 6 * 7, 256)
        self.norm_fc1 = nn.LayerNorm(256)

        self.fc2 = nn.Linear(256, 64)
        self.norm_fc2 = nn.LayerNorm(64)

        # Output heads
        self.policy_head = nn.Linear(64, 7)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.norm_initial(self.conv_initial(x)))

        for block in self.res_blocks:
            x = block(x)

        x = x.view(x.size(0), -1)  # Flatten tensor

        x = F.relu(self.norm_fc1(self.fc1(x)))
        x = F.relu(self.norm_fc2(self.fc2(x)))

        policy = F.softmax(self.policy_head(x), dim=1)
        value = torch.tanh(self.value_head(x))

        return policy, value


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x += residual
        x = F.relu(x)
        return x


class C4Net(nn.Module):
    def __init__(self, num_blocks=3, board_shape=(6, 7)):
        super(C4Net, self).__init__()
        self.conv_initial = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)

        # Create Residual Blocks
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(128, 128) for _ in range(num_blocks)]
        )

        self.fc1 = nn.Linear(128 * np.prod(board_shape), 256)
        self.fc2 = nn.Linear(256, 64)

        # Output heads
        self.policy_head = nn.Linear(64, 7)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv_initial(x))

        # Pass through Residual Blocks
        for block in self.res_blocks:
            x = block(x)

        x = x.view(x.size(0), -1)  # Flatten tensor

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        policy = F.softmax(self.policy_head(x), dim=1)
        value = torch.tanh(self.value_head(x))

        return policy, value


class C4NetLoss(torch.nn.Module):
    def __init__(self):
        super(C4NetLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2
        policy_error = torch.sum((-policy * (1e-8 + y_policy.float()).float().log()), 1)
        total_error = (value_error.view(-1).float() + policy_error).mean()
        return total_error
