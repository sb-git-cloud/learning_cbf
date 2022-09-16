import control
import car
from sklearn import svm, preprocessing
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp as ode
from sklearn.metrics import max_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from scipy.io import loadmat
from scipy.special import expit


if __name__ == '__main__':

    # Load EDF
    data = loadmat('data/sim_track_S.mat')
    edf = data['edf'][0]  # EDF for safe/unsafe region
    Xf = data['Xf']  # x-y position
    xx = data['xx']  # meshgrid x
    yy = data['yy']  # meshgrid y
    X = data['X']  # meshgrid as list

    # Scale data
    X_train, X_test, y_train, y_test = train_test_split(Xf, edf, test_size=0.5, random_state=0)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)
    Xf_transformed = scaler.transform(Xf)
    X_transformed = scaler.transform(X)

    # Set range for hyperparameters
    tuned_parameters = [
        {"gamma": [30], "C": [7], "epsilon": [.1]}
    ]

    # Learn EDF
    reg = GridSearchCV(svm.SVR(), tuned_parameters, cv=10)
    reg.fit(X_train_transformed, y_train)
    model = reg.best_estimator_
    y_pred = model.predict(X_test_transformed)
    me = max_error(y_test, y_pred)

    # Robustness parameter
    alpha = me

    # Plot track and EDF data
    plt.rcParams['figure.figsize'] = [20, 20]
    fig1, ax1 = plt.subplots()  # (subplot_kw={"projection": "3d"})
    Zf = model.predict(Xf_transformed) - alpha
    Zmesh = model.predict(X_transformed) - alpha
    Zmesh = Zmesh.reshape(xx.shape)
    contour = ax1.contour(xx, yy, Zmesh, levels=[0], linewidths=3, colors='r')  # prediction
    sc = ax1.scatter(Xf[:, 0], Xf[:, 1], s=50, c=Zf, linewidth=0)  # data
    plt.colorbar(sc)
    plt.tricontour(Xf[:, 0], Xf[:, 1], edf-1e-5, 0, colors='green')  # data 0 level set


    # Simulation of safety filter
    # ===========================

    # Parameters
    goal = np.array([13, 20])
    car = car.Car()
    max_steering = car.getParams()['max_steering']
    max_steering_rate = car.getParams()['max_steering_rate']
    cbf_gains = {"c1": 4, "c2": 4, "c3": 4}  # class-K functions; set to constants c1, c2, c3
    svm_params = {"b": model.intercept_, "dc": model.dual_coef_, "sv": model.support_vectors_,
                  "gamma": model.get_params()["gamma"], "scaler": scaler}
    x0 = np.array([14, 70, -np.pi * 1 / 3, 0])
    tend = 170  # simulation time

    # Time computation for overriding control
    run_loop = False  # execute additional simple loop and compare time to vectorized version

    # Run continuous-time (CT) sim
    ode_sol_ct = ode(control.ctClosedLoop, [0, tend], x0, args=(car, cbf_gains, svm_params, goal, alpha),
                     method='RK45', max_step=1e-1)

    # Rerun to collect variables
    u_ct = float('inf')*np.ones(ode_sol_ct.t.shape)
    unom_ct = float('inf')*np.ones(ode_sol_ct.t.shape)
    usafe_ct = float('inf')*np.ones(ode_sol_ct.t.shape)
    time_vec_ct = np.zeros(ode_sol_ct.t.shape)
    time_loop_ct = np.zeros(ode_sol_ct.t.shape)
    h0_ct = float('inf') * np.ones(ode_sol_ct.t.shape)
    h2_ct = float('inf') * np.ones(ode_sol_ct.t.shape)
    Lfh1_Lgh1usafe_ch1_ct = float('inf') * np.ones(ode_sol_ct.t.shape)
    Lfh2_Lgh2usafe_ch2_ct = float('inf') * np.ones(ode_sol_ct.t.shape)
    Lfh2_Lgh2unom_ch2_ct = float('inf') * np.ones(ode_sol_ct.t.shape)
    for i in range(len(ode_sol_ct.t)):
        dx, u_ct[i], unom_ct[i], usafe_ct[i], time_vec_ct[i], time_loop_ct[i], Lfh2_Lgh2unom_ch2_ct[i], \
        Lfh2_Lgh2usafe_ch2_ct[i], h2_ct[i], Lfh1_Lgh1usafe_ch1_ct[i], h0_ct[i] = \
            control.ctClosedLoop(ode_sol_ct.t[i], ode_sol_ct.y[:, i], car, cbf_gains, svm_params, goal,
                                       alpha, solver=False, loop=run_loop)

    # Compute  Lfh2_Lgh2u_ch2_ct
    u_equ_unom = np.equal(u_ct, unom_ct)
    u_equ_usafe = np.equal(u_ct, usafe_ct)
    Lfh2_Lgh2u_ch2_ct = Lfh2_Lgh2usafe_ch2_ct * u_equ_usafe + Lfh2_Lgh2unom_ch2_ct * u_equ_unom


    # Plot results
    # ============

    # Trajectory
    ax1.plot(x0[0], x0[1], 'rx', label=r'$x_0$', markersize=15, linewidth=3)
    ax1.plot(ode_sol_ct.y[0, :], ode_sol_ct.y[1, :], color='lime', label='travelled path', linewidth=2)
    ax1.plot(goal[0], goal[1], 'ro', label='Goal', markersize=15)
    plt.legend()

    # Steering angle
    fig2, ax2 = plt.subplots()
    ax2.plot(ode_sol_ct.t, max_steering * (2 / (1 + expit(ode_sol_ct.y[3, :])) - 1), label=r'$\delta$', linewidth=2)
    ax2.plot([0, ode_sol_ct.t[-1]], [max_steering] * 2, label=r'$\delta_{\rm max}$', linewidth=2)
    ax2.plot([0, ode_sol_ct.t[-1]], [-max_steering] * 2, label=r'$\delta_{\rm min}$', linewidth=2)
    plt.legend()

    # Control input
    fig3, ax3 = plt.subplots()
    ax3.plot(ode_sol_ct.t, unom_ct, label=r'$u_{\rm nom}$', linewidth=2)
    ax3.plot(ode_sol_ct.t, usafe_ct, label=r'$u_{\rm safe}$', linewidth=2)
    ax3.plot(ode_sol_ct.t, u_ct, label=r'$u_{\rm or}$', linewidth=2, linestyle='dashed')
    ax3.plot([0, ode_sol_ct.t[-1]], [max_steering_rate] * 2, label=r'$u_{\rm max}$', linewidth=2)
    ax3.plot([0, ode_sol_ct.t[-1]], [-max_steering_rate] * 2, label=r'$u_{\rm min}$', linewidth=2)
    plt.legend()

    # Barriers
    fig4, ax4 = plt.subplots()
    ax4.plot(ode_sol_ct.t, h2_ct, label=r'$h_2$', linewidth=1.5)
    ax4.plot(ode_sol_ct.t, h0_ct, label=r'$h_2$', linewidth=1.5)
    ax4.plot([0, ode_sol_ct.t[-1]], [0] * 2, label='0', linewidth=1, c='black')
    ax4.plot(ode_sol_ct.t, Lfh2_Lgh2u_ch2_ct, label=r'$L_fh_2+L_gh_2u_{\rm or}+\alpha_2(h_2)$', linewidth=3)
    ax4.plot(ode_sol_ct.t, Lfh2_Lgh2usafe_ch2_ct, label=r'$L_fh_2+L_gh_2u_{\rm safe}+\alpha_2(h_2)$', linewidth=1.5,
             linestyle='dashed')
    ax4.plot(ode_sol_ct.t, Lfh2_Lgh2unom_ch2_ct, label=r'$L_fh_2+L_gh_2u_{\rm nom}+\alpha_2(h_2)$', linewidth=1.5,
             linestyle='dashed')
    plt.legend()

    # Computing time
    fig5, ax5 = plt.subplots()
    ax5.semilogy(ode_sol_ct.t, time_vec_ct, label='computing time vec. alg.')
    if run_loop:
        ax5.semilogy(ode_sol_ct.t, time_loop_ct, label='computing time loop alg.')
    plt.legend()
    plt.show()
    
    # Print mean and std. dev.
    print("--- Mean of vectorized computation: %ss with std dev of %ss" % (np.mean(time_vec_ct), np.std(time_vec_ct)))
    if run_loop:
        print("--- Mean of loop computation: %ss with std dev of %ss" % (np.mean(time_loop_ct), np.std(time_loop_ct)))



