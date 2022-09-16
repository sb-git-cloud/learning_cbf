import numpy as np
import time
from scipy.special import expit, logit

def ctClosedLoop(t, x, car, gains, svm_params, goal, alpha, solver=True, loop=False):

    # Compute nominal control
    unom = getNomCntrl(x, goal, car)

    # Compute safe control
    u, usafe, time_vec, time_loop, Lfh2_Lgh2unom_ch2, Lfh2_Lgh2usafe_ch2, h2, Lfh1_Lgh1usafe_ch1, h0 \
        = getUOverride(t, x, unom, car, gains, svm_params, alpha, loop=loop)

    # Dynamics
    dx = car.dxdtKinBicycleFront(t, x, u, car)

    if solver:
        return dx
    else:
        return dx, u, unom, usafe, time_vec, time_loop, Lfh2_Lgh2unom_ch2, Lfh2_Lgh2usafe_ch2, h2, \
               Lfh1_Lgh1usafe_ch1, h0



def getUOverride(t, x, unom, car, gains, svm_params, alpha, loop=False):

    # Compute safety filter
    usafe, Lfh2_Lgh2usafe_ch2, Lfh2_Lgh2unom_ch2, h2, Lfh1_Lgh1usafe_ch1, h0, time_vec, time_loop \
        = getUsafeAndBarriers(t, x, car, svm_params, gains, alpha, unom, vectorized=True, loop=loop)
    start = time.time()
    if Lfh2_Lgh2unom_ch2 <= 0:
        u = usafe
    else:
        u = unom
    end = time.time()
    time_vec += end-start
    time_loop += end-start

    return u, usafe, time_vec, time_loop, Lfh2_Lgh2unom_ch2, Lfh2_Lgh2usafe_ch2, h2, Lfh1_Lgh1usafe_ch1, h0

def getUsafeAndBarriers(t, x, car, svm_params, gains, alpha, unom, vectorized=True, loop=False):

    # Start timer
    start_tot = time.time()

    # Parameters
    vx = car.getParams()["vx"]
    L = car.getParams()["L"]
    c1 = gains["c1"]
    c2 = gains["c2"]
    c3 = gains["c3"]
    max_steering = car.getParams()["max_steering"]
    max_steering_rate = car.getParams()["max_steering_rate"]
    bound = 1e-12  # to avoid zero division
    umin = - max_steering_rate

    # Steering angle
    logsig = expit(x[3])
    delta = max_steering * (2 * logsig - 1)
    global_steering = x[2] + delta


    # Dynamics
    xyd = vx * np.array([np.cos(global_steering), np.sin(global_steering)])
    f2 = vx ** 2 * np.sin(delta) / L * np.array([-np.sin(global_steering), np.cos(global_steering)])
    g2 = vx * np.array([-np.sin(global_steering), np.cos(global_steering)])

    ff2 = -vx**3 / (L**2) * np.sin(delta) ** 2 * np.array([np.cos(global_steering), np.sin(global_steering)])
    gf2 = vx ** 2 / L * (
            np.cos(delta) * np.array([-np.sin(global_steering), np.cos(global_steering)])
            - np.sin(delta) * np.array([np.cos(global_steering), np.sin(global_steering)]))

    fg2 = -vx ** 2 / L * np.sin(delta) * np.array([np.cos(global_steering), np.sin(global_steering)])
    gg2 = - vx * np.array([np.cos(global_steering), np.sin(global_steering)])

    xy = x[:2]
    end_tot = time.time()

    # Compose ICCBFs
    h, g, H, K, time_vec, time_loop = getBarrier(xy, svm_params, alpha, vectorized=vectorized, loop=loop,
                                                 xdot=xyd)
    start_time = time.time()
    h0 = h
    Lfh0 = g @ xyd
    h1 = Lfh0 + c1 * h0

    Lfh1 = xyd.T @ H @ xyd + g @ f2 + c1 * Lfh0
    Lgh1 = g @ g2
    h2 = Lfh1 + np.abs(Lgh1) * umin + c2 * h1

    Lfh2 = 3 * xyd.T @ H @ f2 + np.dot(np.dot(np.dot(xyd, K), xyd), xyd) + (c1 + c2) * (xyd.T @ H @ xyd + g @ f2) \
           + c1 * c2 * g @ xyd + g @ ff2 + np.sign(g @ g2) * (xyd.T @ H @ g2 + g @ fg2) * umin
    Lgh2 = 2 * xyd.T @ H @ g2 + (c1 + c2) * g @ g2 + g @ gf2 + np.sign(g @ g2) * g @ gg2 * umin

    # Compute usafe
    num = -(Lfh2 + c3 * h2)
    den = Lgh2
    if np.abs(den) < bound:
        usafe = num * np.sign(den) / bound
    else:
        usafe = float(num / den)
    if np.abs(usafe) > max_steering_rate:  # Saturate
        usafe = np.sign(usafe) * max_steering_rate

    # Stop timer
    end_time = time.time()

    # ICCBF inequalities
    Lfh1_Lgh1usafe_ch1 = Lfh1 + Lgh1 * usafe + c2 * h1
    Lfh2_Lgh2usafe_ch2 = Lfh2 + Lgh2 * usafe + c3 * h2
    Lfh2_Lgh2unom_ch2 = Lfh2 + Lgh2 * unom + c3 * h2

    # Add time of computing barriers and that of computing usafe
    time_vec += end_time - start_time + end_tot - start_tot
    time_loop += end_time - start_time + end_tot - start_tot

    return float(usafe), Lfh2_Lgh2usafe_ch2[0], Lfh2_Lgh2unom_ch2[0], h2[0], Lfh1_Lgh1usafe_ch1, h0[0], time_vec, time_loop

def getNomCntrl(x, goal, car):
    max_steering = car.getParams()["max_steering"]
    max_steering_rate = car.getParams()["max_steering_rate"]
    sgm = expit(x[3])
    delta = max_steering * (2 * sgm - 1)

    # Going towards goal
    k1 = .1
    k2 = 1
    theta_ref = np.arctan2((goal[1] - x[1]), (goal[0] - x[0]))
    theta = x[2]
    err_head = ((((theta_ref - theta) % 360) + 540) % 360) - 180
    desired_steering = k1 * err_head
    err_steering = ((((desired_steering - delta) % 360) + 540) % 360) - 180
    u = k2 * (err_steering)

    # Saturate
    if np.abs(u) > max_steering_rate:
        u = np.sign(u) * max_steering_rate

    return float(u)

def getBarrier(xy, svm_params, alpha, vectorized=True, loop=False, xdot=[]):


    gamma = svm_params["gamma"]
    b = svm_params["b"]
    dc = svm_params["dc"]
    sv = svm_params["sv"]
    dim_sv = dc.shape[1]
    scaler = svm_params["scaler"]
    xy = scaler.transform([xy])[0]
    C = np.array(1/np.sqrt(scaler.var_))
    time_vec = 0
    time_loop = 0
    if vectorized:
        start_time = time.time()
        Xi_X = sv - xy
        Xi_X_ip = Xi_X ** 2
        exponent = -gamma * np.sum(Xi_X_ip, axis=1)
        Hi = np.exp(exponent)

        hs = dc @ Hi
        h = hs + b - alpha

        # Gradient
        G = np.multiply(Xi_X.T, [Hi, Hi]) * dc
        g = 2 * gamma * np.sum(G, axis=1) * C

        # Hessian
        Hi_dc = (Hi * dc)
        H_non_scale = 2 * gamma * (2 * gamma * Xi_X.T * Hi_dc @ Xi_X - hs * np.eye(2))
        C_outer = np.outer(C, C)
        H = H_non_scale * C_outer


        # Third derivative
        cross_terms = Xi_X[:, 0] * Xi_X[:, 1]
        V2 = np.array([2 * gamma * Xi_X[:, 0] ** 2 - 3, 2 * gamma * cross_terms, 2 * gamma * cross_terms,
                       2 * gamma * Xi_X[:, 1] ** 2 - 3]).T
        temp = Xi_X.T * Hi_dc @ V2
        K_non_scaled = 4 * gamma ** 2 * temp.reshape((2, 2, 2))
        C2_outer = (np.array([C_outer.ravel(), C_outer.ravel()]).T * C).reshape(2, 2, 2)
        K = K_non_scaled * C2_outer
        end_time = time.time()
        time_vec = (end_time - start_time)

    if loop:
        start_time = time.time()
        h2 = 0
        g2 = np.array([0., 0.])
        H2 = np.array([[0., 0.], [0., 0.]])
        K2 = np.array([[[0., 0.], [0., 0.]]])
        for i in range(dim_sv):
            yi = sv[i, :] - xy
            hi = np.exp(-gamma * np.linalg.norm(yi) ** 2)
            h2 += dc[0, i] * hi
            g2 += yi * dc[0, i] * hi
            H2 += hi * dc[0, i] * (2 * gamma * np.outer(yi, yi) - np.eye(2))

            # Third derivative
            if np.any(xdot):
                dt_xTx = np.array([[2 * yi[0] * xdot[0], xdot[0] * yi[1] + yi[0] * xdot[1]],
                                   [xdot[0] * yi[1] + yi[0] * xdot[1], 2 * yi[1] * xdot[1]]])
                K2 += hi * dc[0,i] * (-dt_xTx + (xdot.T @ yi) * (2 * gamma * np.outer(yi, yi) - np.eye(2)))
        h2 += b - alpha
        g2 *= 2 * gamma
        H2 *= 2 * gamma
        K2 *= 4 * gamma**2
        time_loop = (time.time() - start_time)

    return h, g, H, K, time_vec, time_loop