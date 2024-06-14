import matplotlib.pyplot as plt

from torch import torch, cos, sin, tan, pi

import numpy as np

to_rad = pi/180

def getSoilMetalFrictionAngle(entrance_angle):
    return torch.where(entrance_angle < (61*to_rad), 24*to_rad, ((50.5*to_rad) - 0.45 * entrance_angle))


def getForce(entrance_angle, width, depth, friction_angle, soil_cohesion, soil_weight, rho=None):
    '''
    entrance_angle is angle of the tool from the horizontal\n
    width is the width of the tool cutting the soil\n
    depth is the vetical depth of the tool in cm\n
    friction_angle is the angle of repose for the soil, typically around 31 degrees.\n
    soil_cohesion is the cohesional constant for the soil, 0.294 kPa\n
    soil_weight is the unit weight (density) of the soil, 2.0 g/cc\n
    rho is the rupture angle in front of the soil, according to Passive Earth Pressure Theory, 
        dF / dRho = 0
    use getForce to automatically calculate rho with the Passive Earth Pressure Theory assumption.
    All angle values in radians
    '''

    if rho is None:
        rho = getRhoAngles(entrance_angle, width, depth, friction_angle, soil_cohesion, soil_weight)

    # calculate forces from the sides of the wedge

    # calculate side area. Have two angles, and depth. Use basic trigonometry.
    side_area = 0.5 * (depth**2) * (1/tan(entrance_angle) + 1/tan(rho))

    # normal force acting inwards into the wedge.
    # It is the weight, times the soil pressure, times the average depth of the wedge, times area
    side_normal_force = soil_weight*(1-sin(friction_angle))*(1/3)*depth*side_area

    side_friction_force = side_normal_force*tan(friction_angle)

    side_cohesion_force = soil_cohesion * side_area

    # cohesion force on rupture plane (in dirt, opposite of tool)
    rupture_cohesion_force = soil_cohesion * width * depth / sin(rho)

    cohesion = 2*side_cohesion_force + rupture_cohesion_force

    adhesion_force = 0.5*soil_cohesion*width*depth*sin(entrance_angle)

    # total weight of the soil in the failure wedge
    W = soil_weight * width * side_area

    soil_metal_friction_angle = getSoilMetalFrictionAngle(entrance_angle)

    return 0.01 * (1 / sin(entrance_angle + friction_angle + rho + soil_metal_friction_angle)) * (
        -adhesion_force*cos(entrance_angle + friction_angle + rho) + 
         2*side_friction_force*cos(friction_angle) + 
         W*sin(friction_angle+rho) + 
         cohesion*cos(friction_angle))

def getRhoAngles(entrance_angle, width, depths, friction_angle, soil_cohesion, soil_weight):
    
    rho_angles = []

    for d in depths:
        # compute the derivative of the function with multiple values
        precision = 0.01
        x = torch.linspace(10*to_rad, 60*to_rad, int(90/precision), requires_grad = True)
        Y = getForce(entrance_angle, width, d, friction_angle, soil_cohesion, soil_weight, rho=x)
        y = torch.sum(Y)

        y.backward()

        derivs = x.grad.detach().numpy()

        idx = np.argmin(abs(derivs[1:])) + 1  # deriv = 0
        # idx = np.argmin(Y.detach().numpy()[1:])+1 # lowest value
        # idx = np.argmin(abs(Y.detach().numpy()[1:]))+1 # value = 0

        rho_angles.append(x.detach()[idx])

    # print('deriv closest to zero', x.detach()[idx].item(), 'smallest value', x.detach()[idx_min_val].item(), 'value closest to zero', x.detach()[idx_zeroest_val].item())

    # fig, ax = plt.subplots()
    # ax.plot(x.detach(), Y.detach(), label='force')
    # ax.plot(x.detach(), derivs, label='force deriv')
    # ax.set(ylim=[-1000, 1000])
    # ax.legend()

    # plt.show()

    return torch.tensor(rho_angles)


def getRhoAnglesVector(entrance_angle, width, depths, friction_angle, soil_cohesion, soil_weight):
    
    rho_angles = []

    for d in depths:
        # compute the minimum of the function with multiple values
        steps = 4000
        x = torch.linspace(0*to_rad, 60*to_rad, steps, requires_grad = True)
        Y = getForceVector(entrance_angle, width, d, friction_angle, soil_cohesion, soil_weight, rho=x)
        
        # y = torch.sum(Y)
        # y.backward()
        # derivs = x.grad.detach().numpy()
        # idx = np.argmin(abs(derivs[1:])) + 1  # deriv = 0

        idx = np.argmin(np.linalg.norm(Y.detach().numpy()[1:], axis=1))+1 # value = 0

        rho_angles.append(x.detach()[idx])

    return torch.tensor(rho_angles)


def getForceVector(entrance_angle, width, depth, friction_angle, soil_cohesion, soil_weight, rho=None):
    
    if rho is None:
        rho = getRhoAnglesVector(entrance_angle, width, depth, friction_angle, soil_cohesion, soil_weight)
    # calculate forces from the sides of the wedge

    # calculate side area. Have two angles, and depth. Use basic trigonometry.
    side_area = 0.5 * (depth**2) * (1/tan(entrance_angle) + 1/tan(rho))

    # normal force acting inwards into the wedge.
    # It is the weight, times the soil pressure, times the average depth of the wedge, times area
    side_normal_force = soil_weight*(1-sin(friction_angle))*(1/3)*depth*side_area

    side_friction_force = side_normal_force*tan(friction_angle)

    side_cohesion_force = soil_cohesion * side_area

    # cohesion force on rupture plane (in dirt, opposite of tool)
    rupture_cohesion_force = soil_cohesion * width * depth / sin(rho)

    cohesion = 2*side_cohesion_force + rupture_cohesion_force

    adhesion_force = 0.5*soil_cohesion*width*depth*sin(entrance_angle)
    
    # total weight of the soil in the failure wedge
    W = soil_weight * width * side_area

    normal_force_rupture = W / cos(rho)
    rupture_friction_force = normal_force_rupture * tan(friction_angle)

    x = -adhesion_force*cos(entrance_angle) + 2*side_friction_force*cos(rho) + normal_force_rupture*sin(rho) + rupture_friction_force*cos(rho) + cohesion*cos(rho)

    y = W + 2*side_friction_force*sin(rho) + cohesion*sin(rho) + rupture_friction_force*sin(rho) + cohesion*sin(rho) + adhesion_force*sin(entrance_angle) - normal_force_rupture*cos(rho)

    return 0.1 * torch.stack((-x, -y), dim=1)


if __name__ == '__main__':

    force_test = getForce(torch.tensor(40*to_rad), 5, torch.tensor([5]), torch.tensor(31*to_rad), 0.294, 2).item()
    # print("soil force: ", force_test)

    assert abs(force_test - 4.983981) < 0.001

    angle = torch.tensor(90*to_rad)
    depth = torch.linspace(0, 15, 2000)
    width = 20


    print("display all graphs?")
    ask = input()

    forcesVector1 = getForceVector(angle, width, depth, torch.tensor(31*to_rad), 0.294, 2)

    if ask != 'y':
        fig, ax = plt.subplots()

        ax.plot(depth, -forcesVector1[:, 0], label="Soil Reaction Force")
        ax.legend()
        ax.set(xlabel="Depth (cm)", ylabel="Force X (N)", xticks=range(0,16,3), xlim=[0,15])

    else:

        forcesVector2 = getForceVector(angle, width, depth, torch.tensor(31*to_rad), 2.0, 2)
        forcesVector3 = getForceVector(angle, width, depth, torch.tensor(31*to_rad), 0.0, 2)

        forcesVector4 = getForceVector(angle, width, depth, torch.tensor(31*to_rad), 0.294, 3)
        forcesVector5 = getForceVector(angle, width, depth, torch.tensor(31*to_rad), 0.294, 1)

        forcesVector6 = getForceVector(angle, width, depth, torch.tensor(20*to_rad), 0.294, 2)
        forcesVector7 = getForceVector(angle, width, depth, torch.tensor(45*to_rad), 0.294, 2)

        forcesVector8 = getForceVector(angle - 10*to_rad, width, depth, torch.tensor(31*to_rad), 0.294, 2)
        forcesVector9 = getForceVector(angle - 30*to_rad, width, depth, torch.tensor(31*to_rad), 0.294, 2)

        fig, ax = plt.subplots()
        fig.set_size_inches(9, 9)

        ax.plot(depth, -forcesVector1[:, 0], label="Soil Reaction Force")
        ax.plot(depth, -forcesVector2[:, 0], label="Soil Reaction Force, high cohesion")
        ax.plot(depth, -forcesVector3[:, 0], label="Soil Reaction Force, low cohesion")
        ax.legend()
        ax.set(xlabel="Depth (cm)", ylabel="Force X (N)")

        plt.show()

        fig, ax = plt.subplots()
        fig.set_size_inches(9, 9)
        ax.plot(depth, -forcesVector1[:, 0], label="Soil Reaction Force")
        ax.plot(depth, -forcesVector4[:, 0], label="Soil Reaction Force, high density")
        ax.plot(depth, -forcesVector5[:, 0], label="Soil Reaction Force, low density")
        ax.legend()
        ax.set(xlabel="Depth (cm)", ylabel="Force X (N)")

        plt.show()
        fig, ax = plt.subplots()
        fig.set_size_inches(9, 9)
        ax.plot(depth, -forcesVector1[:, 0], label="Soil Reaction Force")
        ax.plot(depth, -forcesVector6[:, 0], label="Soil Reaction Force, low friction")
        ax.plot(depth, -forcesVector7[:, 0], label="Soil Reaction Force, high friction")
        ax.legend()
        ax.set(xlabel="Depth (cm)", ylabel="Force X (N)")

        plt.show()
        fig, ax = plt.subplots()
        fig.set_size_inches(9, 9)
        ax.plot(depth, -forcesVector1[:, 0], label="Soil Reaction Force, 90 degree entrance")
        ax.plot(depth, -forcesVector8[:, 0], label="Soil Reaction Force, 80 degree entrance")
        ax.plot(depth, -forcesVector9[:, 0], label="Soil Reaction Force, 60 degree entrance")
        ax.legend()
        ax.set(xlabel="Depth (cm)", ylabel="Force X (N)")

    plt.show()


