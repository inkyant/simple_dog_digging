import matplotlib.pyplot as plt

import torch

from model import getForce

import matplotlib.animation as animation

if __name__ == "__main__":

    to_rads = torch.pi / 180.0
    
    # simulation params
    steps = 2000
    
    interval = 30
    total_time = 10
    anim_speed = int(steps * interval * 0.001 / total_time)

    # foot params
    entrance_angle = torch.tensor(80 * to_rads)
    width = 6

    # soil params
    friction_angle = torch.tensor(31 * to_rads)
    cohesion = 0.294
    density = 2

    # simulation
    times = torch.linspace(0, total_time, steps)
    depths = (-4)*torch.sqrt((total_time/2)**2 - (times - (total_time/2))**2)

    # calculate forces for all depths.
    forces = getForce(entrance_angle, width, depths, friction_angle, cohesion, density)*-0.01

    fig, ax = plt.subplots()

    line1 = ax.plot(times[0], forces[0], label="Force")[0]
    line2 = ax.plot(times[0], depths[0], label="Depth")[0]
    ax.set(xlim=[0, total_time], ylim=[-20, 5], xlabel='Time [s]', ylabel='Depth [cm]/Force [N]')

    def update(frame):
        frame *= anim_speed
        # update the plots:
        line1.set_xdata(times[:frame])
        line1.set_ydata(forces[:frame])
        line2.set_xdata(times[:frame])
        line2.set_ydata(depths[:frame])
        return (line1, line2)


    ani = animation.FuncAnimation(fig=fig, func=update, frames=int(steps/anim_speed), interval=interval)
    plt.show()

