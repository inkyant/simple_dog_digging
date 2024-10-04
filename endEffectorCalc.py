import numpy as np

import torch
from torch import cos, sin, tan

import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

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

    return pos


def getCalfPosForAngles(angles: torch.Tensor, leg1):
    
    full_transform = getTransformMatrix(angles + torch.pi, leg1)

    pos = full_transform[:, 0:2, 3]

    return pos


if __name__ == '__main__':

    steps = 4000
    angle1 = torch.cat((torch.linspace(0, (1/8) * torch.pi, steps//2), torch.linspace((1/8) * torch.pi, 0, steps//2)))
    angle2 = torch.linspace(0, (3/4) * torch.pi, steps)

    pos = getFootPosForAngles(torch.stack((angle1, angle2), dim=1), 12, 12)
    calf_pos = getCalfPosForAngles(angle1, 12)

    fig = plt.figure()
    fig.set_size_inches(9, 9)
    
    # marking the x-axis and y-axis 
    axis = plt.axes()  
    
    entrance_angle = angle1 + angle2

    dirt_height = -5

    xlim=[-25, 25]

    dirt_intersect_idx = torch.argmin(abs(pos[:(steps//2), 1] - dirt_height)).item()

    def animate(i):

        axis.clear()
        if i > 1:

            dirtX = [xlim[0], pos[dirt_intersect_idx, 0].item()]
            dirtY = [dirt_height, pos[dirt_intersect_idx, 1].item()]

            if (i > dirt_intersect_idx):

                dist_back = torch.tensor([pos[i, 0] - ((pos[i, 1] - dirt_height) / tan(entrance_angle[i]))])

                dirtX = torch.cat((torch.tensor(dirtX), pos[dirt_intersect_idx:i, 0], dist_back, torch.tensor([xlim[1]])))
                dirtY = torch.cat((torch.tensor(dirtY), pos[dirt_intersect_idx:i, 1], torch.tensor([dirt_height, dirt_height])))
            else:
                dirtX.append(xlim[1])
                dirtY.append(dirt_height)

            axis.plot(dirtX, dirtY, color='brown')

            axis.set(xlim=xlim, ylim=[-25, 10])

            axis.plot([pos[i, 0], calf_pos[i, 0]], [pos[i, 1], calf_pos[i, 1]], color='blue', linewidth=4)
            axis.plot([calf_pos[i, 0], 0], [calf_pos[i, 1], 0], color='blue', linewidth=4)

    anim = FuncAnimation(fig, animate, frames = steps, interval = 1, repeat_delay=1000)
    
    plt.show()





