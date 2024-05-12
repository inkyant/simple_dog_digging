import matplotlib.pyplot as plt

import torch

from model import getForce

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

    # simulation
    max_depth = 5
    length = 10
    trajX = torch.linspace(10, 0, steps)
    trajY = (-2*max_depth / length)*torch.sqrt((length/2)**2 - (trajX - (length/2))**2)

    # calculate forces for all depths.
    forces = getForce(entrance_angle, width, trajY, friction_angle, cohesion, density)*-0.01

    fig, ax = plt.subplots()

    line2 = ax.plot(trajX[0], trajY[0], label="Trajectory")[0]
    ax.set(xlim=[-1, length+1], ylim=[-2*max_depth, 2])

    # timing constants
    time = 10

    def update(frame):
        frame *= 2
        # update the plots:
        line2.set_xdata(trajX[:frame])
        line2.set_ydata(trajY[:frame])
        return line2


    ani = animation.FuncAnimation(fig=fig, func=update, frames=int(steps/2), interval=(time*100 / steps))
    plt.show()

