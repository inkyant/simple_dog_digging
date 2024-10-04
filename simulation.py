import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

import torch
from torch import cos, sin, tan

from model import getForceVector, getRhoAnglesVector
from endEffectorCalc import getFootPosForAngles, getCalfPosForAngles

to_rads = torch.pi / 180.0

def getForcesForPath(pos, time, steps, entrance_angle, width, acc=None):

    # new = torch.zeros_like(pos)
    # mask = pos[:, 1] > 0
    # new[mask] = pos[mask]
    # pos = new

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

    # fig, axs = plt.subplots(2, 1)

    # axs[0].plot(pos[:, 0], vel[:, 1], label="Vertical Velocity")[0]
    # axs[0].plot(pos[:, 0], vel[:, 0], label="Horizontal Velocity")[0]
    # axs[0].set(title="Velocity vs. X Position")
    # axs[0].legend()

    # axs[1].plot(pos[:, 0], acc[:, 1], label="Vertical Acceleration")[0]
    # axs[1].plot(pos[:, 0], acc[:, 0], label="Horizontal Acceleration")[0]
    # axs[1].set(title="Acceleration vs. X Position")
    # axs[1].legend()

    # fig.tight_layout()
    # fig.show()

    return force_applied, soil_forces, total_force

def displaySim(pos, soil_forces, force_applied, calf_pos, entrance_angle, dirt_height, len1, len2, steps, dt):

    dirt_intersect_idx = torch.argmin(abs(pos[:(steps//2), 1] - dirt_height)).item()

    # calculate torque with cross product in three dimensions
    c = torch.cat((calf_pos[dirt_intersect_idx:, :], torch.zeros((steps - dirt_intersect_idx, 1))), dim=1) / 100
    p = torch.cat((pos[dirt_intersect_idx:, :], torch.zeros((steps - dirt_intersect_idx, 1))), dim=1) / 100
    f = torch.cat((force_applied[dirt_intersect_idx:, :], torch.zeros((steps - dirt_intersect_idx, 1))), dim=1)
    hip_torque = torch.cross(p, f)
    calf_torque = torch.cross(p - c, f)

    hip_torque = hip_torque.norm(dim=1)
    calf_torque = calf_torque.norm(dim=1)


    fig, axs = plt.subplots(nrows=1, ncols=1)

    axs.plot(pos[dirt_intersect_idx:, 0], pos[dirt_intersect_idx:, 1])[0]
    axs.set_title("Foot Path", fontsize=28)
    fig.set_size_inches(9, 9)
    axs.set_xlabel("X Position (cm)", fontsize="25")
    axs.set_ylabel("Y Position (cm)", fontsize="25")
    axs.set_xticks(ticks=[-20, -17, -14, -11, -8], labels=["0", "3", "6", "9", "12"], fontsize=20)
    axs.set_yticks(ticks=[-6, -11], labels=["0", "-5"], fontsize=20)
    fig.show()
    fig, axs = plt.subplots(nrows=1, ncols=1)


    axs.plot(pos[dirt_intersect_idx:, 0], soil_forces[dirt_intersect_idx:, 1], label="Vertical Force")[0]
    axs.plot(pos[dirt_intersect_idx:, 0], soil_forces[dirt_intersect_idx:, 0], label="Horizontal Force")[0]
    axs.set_title("Soil Force vs. X Position", fontsize=28)
    fig.set_size_inches(9, 9)
    axs.legend(fontsize="20", loc="lower left")
    axs.set_xlabel("X Position (cm)", fontsize="25")
    axs.set_ylabel("Force (N)", fontsize="25")
    axs.set_xticks(ticks=[-20, -17, -14, -11, -8], labels=["0", "3", "6", "9", "12"], fontsize=20)
    axs.set_yticks(ticks=[0, -20], labels=["0", "-20"], fontsize=20)
    fig.show()

    fig, axs = plt.subplots(nrows=1, ncols=1)

    # axs.set_title("Hip Torque vs. X Position", fontsize=28)
    # fig.set_size_inches(9, 9)
    # axs.set_xlabel("X Position (cm)", fontsize="25")
    # axs.set_ylabel("Torque (Nm)", fontsize="25")
    # axs.set_xticks(ticks=[-20, -17, -14, -11, -8], labels=["0", "3", "6", "9", "12"], fontsize=20)
    # axs.set_yticks(ticks=[0, 2], labels=["0", "2"], fontsize=20)
    # fig.show()
    # fig, axs = plt.subplots(nrows=1, ncols=1)

    axs.plot(pos[dirt_intersect_idx:, 0], calf_torque, label="Calf Torque")[0]
    axs.plot(pos[dirt_intersect_idx:, 0], hip_torque, label="Hip Torque")[0]
    axs.set_title("Joint Torques vs. X Position", fontsize=28)
    axs.legend(fontsize="20", loc="lower left")
    fig.set_size_inches(9, 9)
    axs.set_xlabel("X Position (cm)", fontsize="25")
    axs.set_ylabel("Torque (Nm)", fontsize="25")
    axs.set_xticks(ticks=[-20, -17, -14, -11, -8], labels=["0", "3", "6", "9", "12"], fontsize=20)
    axs.set_yticks(ticks=[0, 2], labels=["0", "2"], fontsize=20)
    fig.show()

    fig.tight_layout()

    #### ANIMATION ####

    fig = plt.figure()
    fig.set_size_inches(9, 9)
    axis = plt.axes() 
    
    # scale down force vectors
    scale = 0.5

    xlim = [-(len1+len2+1), len1]

    def animate(i): 

        axis.clear()
        if i > 1:

            dirtX = [xlim[0], pos[dirt_intersect_idx, 0].item()]
            dirtY = [dirt_height, pos[dirt_intersect_idx, 1].item()]

            if (i > dirt_intersect_idx):
                dist_back = torch.tensor([pos[i, 0] - ((pos[i, 1] - dirt_height) / tan(entrance_angle[i]))])

                dirtX = torch.cat((torch.tensor(dirtX), pos[dirt_intersect_idx:i, 0], dist_back, torch.tensor([xlim[1]])))
                dirtY = torch.cat((torch.tensor(dirtY), pos[dirt_intersect_idx:i, 1], torch.tensor([dirt_height, dirt_height])))
            
            
                axis.arrow(pos[i, 0], pos[i, 1], 
                        dx=force_applied[i, 0]*scale, dy=force_applied[i, 1]*scale, 
                        width=0.05, color='black', label='Force Applied')
                
                axis.arrow(pos[i, 0], pos[i, 1], 
                        dx=soil_forces[i, 0]*scale, dy=soil_forces[i, 1]*scale, 
                        width=0.05, color='orange', label='Soil Reaction Force')
                
                axis.legend(loc='upper left', fontsize=20)
                
            else:
                dirtX.append(xlim[1])
                dirtY.append(dirt_height)

            axis.plot(dirtX, dirtY, color='brown')

            axis.set(xlim=xlim, ylim=[-(len1+1), 10])

            axis.plot([pos[i, 0], calf_pos[i, 0]], [pos[i, 1], calf_pos[i, 1]], color='blue', linewidth=4)
            axis.plot([calf_pos[i, 0], 0], [calf_pos[i, 1], 0], color='blue', linewidth=4)

            axis.set_xticks(ticks=[])
            axis.set_yticks(ticks=[])

            axis.set_xlabel("X Position (cm)", fontsize="25")
            axis.set_ylabel("Y Position (cm)", fontsize="25")
            axis.set_title(f"Simulation Display", fontsize=28)


    animate(1*400)
    fig.show()

    fig = plt.figure()
    fig.set_size_inches(9, 9)
    axis = plt.axes() 
    animate(6*400)
    fig.show()

    fig = plt.figure()
    fig.set_size_inches(9, 9)
    axis = plt.axes() 
    animate(9*400)
    fig.show()

    # anim = FuncAnimation(fig, animate, frames = steps, interval = dt * 1000, repeat_delay=1000)
    
    # plt.show()


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


    # angle1 = torch.cat((torch.linspace(0, torch.pi / 8, steps // 2), torch.linspace(torch.pi / 8, 0, steps // 2)))
    # angle2 = torch.linspace(0, 3 * torch.pi / 4, steps)

    # angle1 = torch.cat((torch.linspace(-torch.pi / 8, 0, steps // 2), torch.linspace(0, -torch.pi / 8, steps // 2)))
    # angle2 = torch.linspace(torch.pi / 2, 3 * torch.pi / 4, steps)

    angle1 = torch.linspace(0, 0, steps)
    angle2 = torch.linspace(torch.pi / 6, 5 * torch.pi / 8, steps)

    leg_len = 11

    pos = getFootPosForAngles(torch.stack((angle1, angle2), dim=1), leg_len, leg_len)
    calf_pos = getCalfPosForAngles(angle1, leg_len)
    acc = None

    entrance_angle = angle1 + angle2
    dirt_pos = -6


    f_applied, f_soil, f_total = getForcesForPath(pos - dirt_pos, time, steps, entrance_angle, 5, acc=acc)

    displaySim(pos, f_soil, f_applied, calf_pos, entrance_angle, dirt_pos, leg_len, leg_len, steps, dt)

