import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

import torch
from torch import tan

from model import getForce, getForceVector, getRhoAngles

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

    # manual setting of acceleration
    acc = torch.stack((torch.zeros(steps), torch.ones(steps) * 0.4), dim=1)

    vel = torch.cat((vel[0, :].reshape((1,2)), vel), dim=0)

    # soil params
    friction_angle = torch.tensor(31 * to_rads)
    cohesion = 0.294
    density = 2
    rho_angles = getRhoAngles(entrance_angle, width, trajY, friction_angle, cohesion, density)
    mass = density * width * 0.5 * (trajY**2) * (1/tan(entrance_angle) + 1/tan(rho_angles))

    # calculate forces for all depths.
    soil_forces = getForceVector(entrance_angle, width, trajY, friction_angle, cohesion, density, rho=rho_angles)

    # ΣF = ma, so F_a - F_s = ma, so F_a = ma + F_s
    force_applied = torch.stack((mass, mass), dim=1) * acc + soil_forces

    fig, ((ax1, ax4), (ax2, ax5), (ax3, ax6)) = plt.subplots(nrows=3, ncols=2)

    ax1.plot(pos[:, 0], pos[:, 1])[0]
    ax1.set(xlim=[-1, length+1], ylim=[-2*max_depth, 2], title="Position")

    ax2.plot(pos[:, 0], vel[:, 1])[0]
    ax2.set(title="Y Velocity vs. X Position")

    ax3.plot(pos[:, 0], acc[:, 1])[0]
    ax3.set(title="Y Accel vs. X Position", ylim=[0,1])

    ax4.plot(pos[:, 0], force_applied[:, 0])[0]
    ax4.set(title="X Force Applied vs. X Position")

    ax5.plot(pos[:, 0], force_applied[:, 1])[0]
    ax5.set(title="Y Force Applied vs. X Position")

    ax6.plot(pos[:, 0], soil_forces[:, 0])[0]
    ax6.set(title="X Soil Force vs. X Position")

    fig.tight_layout()

    plt.show()

    #### ANIMATION ####

    fig = plt.figure()  
    
    # marking the x-axis and y-axis 
    axis = plt.axes(xlim =(0, 10),  
                    ylim =(-10, 1))  
    
    # initializing a line variable 
    line, = axis.plot([], [], lw = 3)  
    line.set_data([], []) 

    width = 0.2
    offset = width / 2
    
    def animate(i): 
        
        axis.clear()
        axis.plot(pos[:i, 0], pos[:i, 1])
        axis.set(xlim=[0, length], ylim=[-2*max_depth, 1])
        
        center = (pos[i, 0].item(), pos[i, 1].item())

        axis.add_patch(Rectangle((pos[i, 0] - offset, pos[i, 1] - offset), width, 10, angle=entrance_angle, rotation_point=center))
        
        scale = 0.1
        
        axis.arrow(center[0], center[1], dx=force_applied[i, 0]*scale, dy=force_applied[i, 1]*scale, width=0.05, color='black')
        axis.arrow(center[0], center[1], dx=soil_forces[i, 0]*scale, dy=soil_forces[i, 1]*scale, width=0.05, color='brown')


    anim = FuncAnimation(fig, animate, frames = steps, interval = dt * 1000, repeat_delay=1000) 
    
    plt.show()
    print("save animation as mp4? [y/n]")
    ask = input()
    if ask == 'Y' or ask == 'y':
        print("saving...")
        anim.save('animation.mp4', writer = 'ffmpeg', fps = 1/dt) 
        print("video file saved.")
