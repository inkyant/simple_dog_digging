import matplotlib.pyplot as plt

from torch import torch, cos, sin, tan, pi

import numpy as np

to_rad = (pi/180)

def getForceWithRho(entrance_angle, width, depth, friction_angle, soil_cohesion, soil_weight, rho):
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


    soil_metal_friction_angle = torch.where(entrance_angle < (61*to_rad), 24*to_rad, (50.5 - 0.45 * entrance_angle)*to_rad)

    return (1 / sin(entrance_angle + friction_angle + rho + soil_metal_friction_angle)) * (
        -adhesion_force*cos(entrance_angle + friction_angle + rho) + 
         2*side_friction_force*cos(friction_angle) + 
         W*sin(friction_angle+rho) + 
         cohesion*cos(friction_angle))



def getForce(entrance_angle, width, depth, friction_angle, soil_cohesion, soil_weight):
    '''
    entrance_angle is angle of the tool from the horizontal\n
    width is the width of the tool cutting the soil\n
    depth is the vetical depth of the tool in cm\n
    friction_angle is the angle of repose for the soil, typically around 31 degrees.\n
    soil_cohesion is the cohesional constant for the soil, 0.294 kPa\n
    soil_weight is the unit weight (density) of the soil, 2.0 g/cc\n
    All angle values in radians
    '''
    # compute the derivative of the function with multiple values
    precision = 2000
    x = torch.linspace(0, pi/2, precision, requires_grad = True)

    Y = getForceWithRho(entrance_angle, width, depth, friction_angle, soil_cohesion, soil_weight, x)
    y = torch.sum(Y)

    y.backward()

    derivs = x.grad.detach().numpy()

    idx = np.argmin(abs(derivs[1:])) + 1

    rho = x.detach().numpy()[idx]

    return getForceWithRho(entrance_angle, width, depth, friction_angle, soil_cohesion, soil_weight, torch.tensor(rho))


if __name__ == '__main__':

    force_test = getForce(torch.tensor(40*to_rad), 5, 5, torch.tensor(31*to_rad), 0.294, 2).item()
    # print("soil force: ", force_test)

    assert abs(force_test - 498.3981) < 0.001

    precision = 2000

    angle = torch.tensor(90)
    depth = torch.linspace(0, 15, precision)
    width = 20

    forces = getForce(angle, width, depth, torch.tensor(30*to_rad), 0.294, 2)

    plt.plot(depth, forces*0.01)
    plt.xlabel("Depth (cm)")
    plt.ylabel("Force")

    plt.show()

