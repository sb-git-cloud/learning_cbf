import numpy as np
from scipy.special import expit
class Car:

    def __init__(self):
        vx = 1
        lf = .15
        lr = .15
        L = lf + lr
        max_steering = 20 * np.pi / 180
        max_steering_rate = np.pi
        self.car_params = {"vx": vx, "L": L, "max_steering": max_steering, "max_steering_rate": max_steering_rate}


    def getParams(self):
        return self.car_params

    def dxdtKinBicycleFront(self, t, x, u, car):

        vx = self.car_params["vx"]
        L = self.car_params["L"]
        max_steering = car.getParams()["max_steering"]

        # Compute steering angle
        logsig = expit(x[3])
        delta = max_steering * (2 * logsig - 1)

        # Compute (dphi)^-1 for scaling control input
        dphi = max_steering * 2 * logsig * (1 - logsig)
        bound = 1e-12
        if np.abs(dphi) <= bound:
            dphi = bound
        uzeta = u / dphi

        # Dynamics
        dx = np.zeros(4)
        dx[0] = np.cos(x[2] + delta) * vx
        dx[1] = np.sin(x[2] + delta) * vx
        dx[2] = np.sin(delta) * vx / L
        dx[3] = uzeta
        return dx

