import matplotlib.pyplot as plt

import torch

from model import getForce, getForceVector

import matplotlib.animation as animation

if __name__ == "__main__":

    to_rads = torch.pi / 180.0
    
    # simulation params
    steps = 2000

    # foot params
    entrance_angle = torch.tensor(80 * to_rads)
    width = 6

    # soil params
    friction_angle = torch.tensor(31 * to_rads)
    cohesion = 0.294
    density = 2

    # digging depth and length
    max_depth = 5
    length = 10

    # simulation
    time = 10
    dt = time / steps


    trajX = torch.linspace(0, 10, steps)
    trajY = (-2*max_depth / length)*torch.sqrt((length/2)**2 - (trajX - (length/2))**2)

    pos = torch.stack((trajX, trajY), dim=1)
    vel = torch.diff(pos, dim=0) / dt
    vel = torch.cat((torch.tensor([[0, -40]]), vel), dim=0)

    acc = torch.diff(vel, dim=0) / dt
    acc = torch.cat((torch.tensor([[0, 0]]), acc), dim=0)

    # calculate forces for all depths.
    forces = getForceVector(entrance_angle, width, trajY, friction_angle, cohesion, density)

    fig, ((ax1, ax4), (ax2, ax5), (ax3, ax6)) = plt.subplots(nrows=3, ncols=2)

    ax1.plot(pos[:, 0], pos[:, 1])[0]
    ax1.set(xlim=[-1, length+1], ylim=[-2*max_depth, 2], title="Position")

    ax2.plot(pos[:, 0], vel[:, 1])[0]
    ax2.set(title="Y Velocity vs. Position")

    ax3.plot(pos[:, 0], acc[:, 1])[0]
    ax3.set(title="Y Accel vs. Position")

    ax4.plot(pos[:, 0], vel[:, 0])[0]
    ax4.set(title="X Velocity vs. Position")

    ax5.plot(pos[:, 0], acc[:, 0])[0]
    ax5.set(title="X Accel vs. Position")

    ax6.plot(pos[:, 0], forces[:, 0])[0]
    ax6.set(title="Force vs. Position")

    plt.show()

