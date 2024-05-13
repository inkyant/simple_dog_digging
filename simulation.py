import matplotlib.pyplot as plt

import torch
from torch import tan

from model import getForce, getForceVector, getRhoAngles

import matplotlib.animation as animation

if __name__ == "__main__":

    to_rads = torch.pi / 180.0
    
    # simulation params
    steps = 2000

    # foot params
    entrance_angle = torch.tensor(80 * to_rads)
    width = 6

    ### foot digging trajectory
    # 10 second trajectory, constant x velocity
    dt = 10 / steps
    max_depth = 5
    length = 10

    trajX = torch.linspace(0, 10, steps)
    # along semi circle
    # trajY = (-2*max_depth / length)*torch.sqrt((length/2)**2 - (trajX - (length/2))**2)
    # along parabola
    trajY = (4*max_depth/(length**2)) * trajX * (trajX - length)

    pos = torch.stack((trajX, trajY), dim=1)
    vel = torch.diff(pos, dim=0) / dt

    acc = torch.diff(vel, dim=0) / dt
    acc = torch.cat((acc[0, :].reshape((1,2)), acc[0, :].reshape((1,2)), acc), dim=0)

    vel = torch.cat((vel[0, :].reshape((1,2)), vel), dim=0)

    # soil params
    friction_angle = torch.tensor(31 * to_rads)
    cohesion = 0.294
    density = 2
    rho_angles = getRhoAngles(entrance_angle, width, trajY, friction_angle, cohesion, density)
    mass = density * width * 0.5 * (trajY**2) * (1/tan(entrance_angle) + 1/tan(rho_angles))

    # calculate forces for all depths.
    soil_forces = getForceVector(entrance_angle, width, trajY, friction_angle, cohesion, density, rho=rho_angles)

    # F = ma, so F_a - F_s = ma, so F_a = ma + F_s
    force_applied = torch.stack((mass, mass), dim=1) * acc + soil_forces

    fig, ((ax1, ax4), (ax2, ax5), (ax3, ax6)) = plt.subplots(nrows=3, ncols=2)

    ax1.plot(pos[:, 0], pos[:, 1])[0]
    ax1.set(xlim=[-1, length+1], ylim=[-2*max_depth, 2], title="Position")

    ax2.plot(pos[:, 0], vel[:, 1])[0]
    ax2.set(title="Y Velocity vs. Position")

    ax3.plot(pos[:, 0], acc[:, 1])[0]
    ax3.set(title="Y Accel vs. Position", ylim=[-5,5])

    ax4.plot(pos[:, 0], force_applied[:, 0])[0]
    ax4.set(title="Y Force Applied vs. Position")

    ax5.plot(pos[:, 0], force_applied[:, 0])[0]
    ax5.set(title="X Force Applied vs. Position")

    ax6.plot(pos[:, 0], soil_forces[:, 0])[0]
    ax6.set(title="X Soil Force vs. Position")

    plt.show()

