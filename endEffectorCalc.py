import numpy as np

import torch
from torch import cos, sin

import matplotlib.pyplot as plt

# using denavit-hartenberg convention

# Create the homogeneous transformation matrix
def getTransformMatrix(angles: torch.Tensor, leg_length):

    cos_t = cos(angles)
    sin_t = sin(angles)

    s = angles.size(0)

    row1 = torch.stack((cos_t, -sin_t, torch.zeros((s)), leg_length * cos_t), dim=1).unsqueeze(1)
    row2 = torch.stack((sin_t,  cos_t, torch.zeros((s)), leg_length * sin_t), dim=1).unsqueeze(1)
    row3_4 = torch.tensor([[0, 0, 1, 0], [0, 0, 0, 1]]).expand((s, -1, -1))

    transform = torch.cat((row1, row2, row3_4), dim=1)

    return transform


def getFootPosForAngles(angles: torch.Tensor, leg1, leg2):
    
    full_transform = getTransformMatrix(angles[:, 0] + torch.pi, leg1) @ getTransformMatrix(angles[:, 1], leg2)

    pos = full_transform[:, 0:2, 3]

    start_pos = pos[0].expand(pos.size(0), -1)

    return pos - start_pos


if __name__ == '__main__':

    steps = 4000
    angle1 = torch.cat((torch.linspace(0, torch.pi / 8, steps // 2), torch.linspace(torch.pi / 8, 0, steps // 2)))
    angle2 = torch.linspace(torch.pi / 4, 3 * torch.pi / 4, steps)

    pos = getFootPosForAngles(torch.stack((angle1, angle2), dim=1), 12, 12)

    print(pos, "\n size:", pos.shape)

    c = torch.linspace(0, 1,  steps)
    c[(steps//2 - 10):(steps//2 + 10)] = 1

    plt.scatter(pos[:, 0], pos[:, 1], c=c)

    plt.show()




