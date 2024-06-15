import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

import torch
from torch import cos, sin, tan

from model import getForceVector, getRhoAnglesVector, getSoilMetalFrictionAngle
from endEffectorCalc import getFootPosForAngles

to_rads = torch.pi / 180.0

def getForcesForPath(pos, time, steps, entrance_angle, width, acc=None):

    dt = time / steps

    vel = torch.diff(pos, dim=0) / dt

    if acc is None:
        print("acc not specfied, calculating...")
        acc = torch.diff(vel, dim=0) / dt
        # fill in by just assuming starting accel, also switch from cm/s^2 to m / s^2
        acc = 0.01 * torch.cat((acc[0, :].reshape((1,2)), acc[0, :].reshape((1,2)), acc), dim=0)

    vel = torch.cat((vel[0, :].reshape((1,2)), vel), dim=0)

    # soil params
    friction_angle = torch.tensor(31 * to_rads)
    cohesion = 0.294
    density = 2
    rho_angles = getRhoAnglesVector(entrance_angle, width, pos[:, 1], friction_angle, cohesion, density)

    max_depth_idx = torch.argmax(abs(pos[:, 1]))
    dirt_dug_through = torch.cat((pos[:max_depth_idx, 1], torch.ones(steps - max_depth_idx)*pos[max_depth_idx, 1]))
    
    mass = 0.001 * density * width * 0.5 * (dirt_dug_through**2) * (1/tan(entrance_angle) + 1/tan(rho_angles))

    # calculate forces for all depths.
    soil_forces = getForceVector(entrance_angle, width, pos[:, 1], 
                                 friction_angle, cohesion, density, rho=rho_angles)

    # ΣF = ma, so F_a + F_s = ma, so F_a = ma - F_s
    total_force = torch.stack((mass, mass), dim=1) * acc
    force_applied = total_force - soil_forces

    fig, axs = plt.subplots(2, 1)

    axs[0].plot(pos[:, 0], vel[:, 1], label="Vertical Velocity")[0]
    axs[0].plot(pos[:, 0], vel[:, 0], label="Horizontal Velocity")[0]
    axs[0].set(title="Velocity vs. X Position")
    axs[0].legend()

    axs[1].plot(pos[:, 0], acc[:, 1], label="Vertical Acceleration")[0]
    axs[1].plot(pos[:, 0], acc[:, 0], label="Horizontal Acceleration")[0]
    axs[1].set(title="Acceleration vs. X Position")
    axs[1].legend()

    fig.tight_layout()
    fig.show()

    return force_applied, soil_forces, total_force

def displaySim(pos, soil_forces, force_applied, total_force, entrance_angle, steps, dt):

    max_depth = torch.min(pos[:, 1]).item()
    length = pos[steps-1, 0]

    fig, axs = plt.subplots(nrows=2, ncols=1)

    axs[0].plot(pos[:, 0], pos[:, 1])[0]
    axs[0].set(xlim=[-1, length+1], ylim=[2*max_depth, 2], title="Position")

    axs[1].plot(pos[:, 0], soil_forces[:, 1], label="Vertical Force")[0]
    axs[1].plot(pos[:, 0], soil_forces[:, 0], label="Horizontal Force")[0]
    axs[1].set(title="Soil Force vs. X Position")
    axs[1].legend()

    fig.tight_layout()

    #### ANIMATION ####

    fig = plt.figure()  
    fig.set_size_inches(9, 9)
    
    # marking the x-axis and y-axis 
    axis = plt.axes()  
    
    # initializing a line variable 
    line, = axis.plot([], [], lw = 3)  
    line.set_data([], []) 

    # leg display width
    width = 0.2
    offset = width / 2
    
    # scale down force vectors
    scale = 0.1

    def animate(i): 

        axis.clear()
        if i > 1:

            dirt_beforeX = torch.tensor((-1, 0))

            dist_back = pos[i, 1].item() * (1/tan(entrance_angle))

            dirt_extendX = torch.tensor((pos[i, 0].item() + dist_back, length+1))
            dirt_extendY = torch.tensor((0, 0))

            axis.plot(torch.cat((dirt_beforeX, pos[:i, 0], dirt_extendX)), 
                      torch.cat((dirt_extendY, pos[:i, 1], dirt_extendY)), color='brown')
            axis.set(xlim=[-1, length+1], ylim=[2*max_depth, 1])
        
            center = (pos[i, 0].item(), pos[i, 1].item())

            axis.add_patch(Rectangle((pos[i, 0] - offset, pos[i, 1] - offset), width, 10, angle=90-entrance_angle*(1/to_rads), rotation_point=center))
            
            axis.arrow(center[0], center[1], 
                    dx=force_applied[i, 0]*scale, dy=force_applied[i, 1]*scale, 
                    width=0.05, color='black', label='Force Applied')
            
            axis.arrow(center[0], center[1], 
                    dx=soil_forces[i, 0]*scale, dy=soil_forces[i, 1]*scale, 
                    width=0.05, color='orange', label='Soil Reaction Force')
            
            # axis.arrow(center[0], center[1], 
            #         dx=total_force[i, 0]*scale, dy=total_force[i, 1]*scale, 
            #         width=0.05, color='green', label='Force Total')
            
            # axis.set_xticks(ticks=[])
            # axis.set_yticks(ticks=[])

            axis.set_xlabel("X Position", fontsize="20")
            axis.set_ylabel("Y Position", fontsize="20")
            
            # axis.arrow(center[0], center[1], 
            #         dx=torch.cos(movement_angle[i]), dy=torch.sin(movement_angle[i]), 
            #         width=0.05, color='red', label='Angle')

            axis.legend(loc='lower left', fontsize=20)


    anim = FuncAnimation(fig, animate, frames = steps, interval = dt * 1000, repeat_delay=1000)
    
    plt.show()


    ask = 'y' # set to 'y' to prompt for saving video

    if ask == 'y':
        print("save animation as mp4? [y/n]")
        ask = input()
        if ask == 'Y' or ask == 'y':
            print("saving...")
            anim.save('animation.mp4', writer = 'ffmpeg', fps = 1/dt)
            print("video file saved.")


if __name__ == "__main__":

    # sim params
    steps = 4000
    time = 10
    dt = time / steps

    # # foot params
    entrance_angle = torch.tensor(80*to_rads)
    width = 5

    ### foot digging trajectory
    max_depth = 5
    length = 10

    trajX = torch.linspace(0, 10, steps)
    # along semi circle
    trajY = (-2*max_depth / length)*torch.sqrt((length/2)**2 - (trajX - (length/2))**2)
    # along parabola
    # trajY = (4*max_depth/(length**2)) * trajX * (trajX - length)

    pos = torch.stack((trajX, trajY), dim=1)

    acc = torch.stack((torch.zeros(steps), torch.ones(steps) * 0.4), dim=1)

    if True:
        angle1 = torch.linspace(0, 0, steps)
        angle2 = torch.linspace(0, torch.pi / 2, steps)

        pos = getFootPosForAngles(torch.stack((angle1, angle2), dim=1), 2, 2)
        acc = None
    
    if True:
        angle1 = torch.cat((torch.linspace(0, torch.pi / 8, steps // 2), torch.linspace(torch.pi / 8, 0, steps // 2)))
        angle2 = torch.linspace(torch.pi / 4, 3 * torch.pi / 4, steps)

        pos = getFootPosForAngles(torch.stack((angle1, angle2), dim=1), 12, 12)
        acc = None


    f_applied, f_soil, f_total = getForcesForPath(pos, time, steps, entrance_angle, width, acc=acc)


    displaySim(pos, f_soil, f_applied, f_total, entrance_angle, steps, dt)

