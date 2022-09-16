#!/usr/bin/python3
# /usr/bin/env python

import numpy as np
import time
from joblib import load
import sklearn
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from tf.transformations import euler_from_quaternion
from std_msgs.msg import Float32, Int16
from scipy.special import expit, logit
from cbf_pkg.msg import CbfFilterData

POSE_TOPIC_NAME = '/amcl_pose'
CMD_VEL_PUB_TOPIC = '/cmd_vel'
NODE_NAME = 'cbf_safety_filter'
FILTER_TOPIC_NAME = '/cbf/filterData'


class CbfSafetyFilter:
    def __init__(self):
    
        rospy.init_node(NODE_NAME, anonymous=False)
        # Subscriber/publisher
        self.pose_sub = rospy.Subscriber(POSE_TOPIC_NAME, PoseWithCovarianceStamped, self.cbPose)
        self.vel_pub = rospy.Publisher(CMD_VEL_PUB_TOPIC, Twist, queue_size=1)
        self.cbf_control_pub = rospy.Publisher(FILTER_TOPIC_NAME, CbfFilterData, queue_size=1)

        # Actuator and CBF values
        self.beta = rospy.get_param('/cbf/beta', .5)  # offset of EDF (\delta in paper)
        self.alpha0 = rospy.get_param('/cbf/alpha0', 1)  # CBF gain
        self.alpha1 = rospy.get_param('/cbf/alpha1', 1)  # CBF gain
        self.alpha2 = rospy.get_param('/cbf/alpha2', 1)  # CBF gain
        self.Ts = rospy.get_param('/cbf/Ts', .01)
        self.observe_computation_time = rospy.get_param('/cbf/observe_computation_time', False)  # Log computing time 
        self.svm_model_name = rospy.get_param('/cbf/svm_model_name')  # Model parameter
        self.svm_scaler_name = rospy.get_param('/cbf/svm_scaler_name')  # Scaler for SVM input data
        self.nom_steering = rospy.get_param('/cbf/nom_steering_cmd', 0)  # Nominal steering command
        self.max_steering = rospy.get_param('/cbf/max_steering', 30*180/np.pi)
        self.max_steering_rate = rospy.get_param('/cbf/max_steering_rate', np.pi)
        self.zero_bound = float(rospy.get_param('/cbf/zero_bound', 1e-10))  # bound to avoid zero division 
        self.car_length = rospy.get_param('/cbf/car_length', 1)

        # Convert desired speed to throttle cmd
        self.desired_speed = rospy.get_param('/cbf/desired_speed')  # constant
        self.vel2throttle_off = rospy.get_param('/cbf/vel2throttle_off')  # coefficients determined by experiment (see Matlab)
        self.vel2throttle_grad = rospy.get_param('/cbf/vel2throttle_grad')
        self.throttle_cmd =  (self.desired_speed-self.vel2throttle_off) / self.vel2throttle_grad
        

        # Initilize class variables
        self.x = np.zeros(4)  # State vector
        self.svm_model = load(self.svm_model_name)  # trained SVM model
        self.scaler = load(self.svm_scaler_name)
        self.svm_params = { "b": self.svm_model.intercept_,
                            "dc": self.svm_model.dual_coef_,
                            "sv": self.svm_model.support_vectors_,
                            "gamma": self.svm_model.get_params()["gamma"],
                            "scaler": self.scaler}  # SVM parameters

        self.x_pose_buffer = 0.
        self.y_pose_buffer = 0.
        self.yaw_pose_buffer = 0.

        self.xf_pose_buffer = 0.
        self.yf_pose_buffer = 0.
        self.zeta_buffer = 0.
        self.unom = 0.

        # end __init__

    def cbPose(self, pose_data):
        # car orientation
        quaternion = (pose_data.pose.pose.orientation.x, pose_data.pose.pose.orientation.y, pose_data.pose.pose.orientation.z, pose_data.pose.pose.orientation.w)
        euler = euler_from_quaternion(quaternion)
        self.yaw_pose_buffer = euler[2]

        # car coordinates
        self.x_pose_buffer = pose_data.pose.pose.position.x
        self.y_pose_buffer = pose_data.pose.pose.position.y

        # front axle car (car coordinates are for rear axle)
        self.xf_pose_buffer = self.x_pose_buffer + np.cos(self.yaw_pose_buffer) * self.car_length
        self.yf_pose_buffer = self.y_pose_buffer + np.sin(self.yaw_pose_buffer) * self.car_length
        
        # rospy.loginfo('X-pos: %f; Xf-pos: %f; Y-pos: %f; Yf-pos: %f', self.x_pose_buffer, self.xf_pose_buffer, self.y_pose_buffer, self.yf_pose_buffer)

    def updateStates(self):

        # state vector
        self.x[0] = self.xf_pose_buffer
        self.x[1] = self.yf_pose_buffer
        self.x[2] = self.yaw_pose_buffer
        self.x[3] = self.zeta_buffer

    def filter(self):
        # This is the main function that computes the steering command depending on the barrier and the nominal control.

        # Get latest measurement
        self.updateStates()
        # rospy.loginfo('X-pos: %f; Y-pos: %f', self.x_pose_buffer, self.y_pose_buffer)

        self.updateNomSteering()

        # Get ICCBF variables
        usafe, h2, h1, h0, Lfh2_Lgh2usafe_ch2, Lfh2_Lgh2unom_ch2, Lfh1_Lgh1usafe_ch1, Lfh1_Lgh1unom_ch1, time_vec \
                = self.getUSafeAndIccbfs()

        # Overwrite only if nominal control is unsafe
        if Lfh2_Lgh2unom_ch2 <= 0:
            u = usafe
        else:
            u = self.unom

        # Change of coordinates
        zeta = self.x[3]
        smoid = expit(zeta)
        dphi = self.max_steering * 2 * smoid * (1 - smoid)
        if np.abs(dphi) <= self.zero_bound:
            dzeta = u * np.sign(dphi) / self.zero_bound
        else:
            dzeta = u/dphi

        # Update steering_angle and zeta
        self.zeta_buffer += dzeta*self.Ts
        smoid_kplus1 = expit(self.zeta_buffer)
        delta = self.max_steering * (2 * smoid_kplus1 - 1)
        steering_cmd = delta

        # Log data
        rospy.loginfo('Nominal steering (rate): %f; Safe steering (rate): %f; Implemented steering: %f; h0: %f; Lfh2_Lgh2unom_ch2: %f, Lfh2_Lgh2usafe: %f', u, usafe, steering_cmd, h0, Lfh2_Lgh2unom_ch2, Lfh2_Lgh2usafe_ch2)

        # Velocity command
        move_cmd = Twist()
        move_cmd.linear.x = self.throttle_cmd
        move_cmd.angular.z = -steering_cmd # MINUS as vesc defines angle in opposite direction

        # Publish values
        self.vel_pub.publish(move_cmd)

        # Publish values for recording
        cbf_data = CbfFilterData()
        cbf_data.header.stamp = rospy.Time.now()
        cbf_data.u_override = u
        cbf_data.u_safe = usafe
        cbf_data.u_nom = self.unom
        cbf_data.steering_angle = steering_cmd
        cbf_data.h2 = h2
        cbf_data.h1 = h1
        cbf_data.h0 = h0
        cbf_data.Lfh2_Lgh2usafe_ch2 = Lfh2_Lgh2usafe_ch2
        cbf_data.Lfh2_Lgh2unom_ch2 = Lfh2_Lgh2unom_ch2
        cbf_data.Lfh1_Lgh1usafe_ch1 = Lfh1_Lgh1usafe_ch1
        cbf_data.Lfh1_Lgh1unom_ch1 = Lfh1_Lgh1unom_ch1

        self.cbf_control_pub.publish(cbf_data)

    def updateNomSteering(self):
        # Change of coordinates
        zeta = self.x[3]
        smoid = expit(zeta)
        delta = self.max_steering * (2 * smoid - 1)
        self.unom = .5*(self.nom_steering-delta)/self.Ts
        
        if abs(self.unom) > self.max_steering_rate:
            self.unom = np.sign(self.unom) * self.max_steering_rate


    def getUSafeAndIccbfs(self):
        
        # Avoid division by zero
        bound_ = self.zero_bound

        # Convert zeta to delta as delta = phi(zeta)
        zeta = self.x[3]
        smoid = expit(zeta)
        delta = self.max_steering * (2 * smoid - 1)

        # Parameters
        L = self.car_length
        vf = self.desired_speed
        umin_ = - self.max_steering_rate

        # First order dyanmics
        global_steering = self.x[2] + delta
        xyd = vf * np.array([np.cos(global_steering), np.sin(global_steering)])

        # Second order
        f2 = vf ** 2 * np.sin(delta) / L * np.array([-np.sin(global_steering), np.cos(global_steering)])
        g2 = vf * np.array([-np.sin(global_steering), np.cos(global_steering)])

        # Third order related to f2
        ff2 = -vf**3 / L**2 * np.sin(delta)**2 * np.array([np.cos(global_steering), np.sin(global_steering)])
        gf2 = vf**2 / L * (
                np.cos(delta)*np.array([-np.sin(global_steering), np.cos(global_steering)])
                - np.sin(delta)*np.array([np.cos(global_steering), np.sin(global_steering)]))

        # Third order related to g2
        fg2 = -vf**2/L*np.sin(delta)*np.array([np.cos(global_steering), np.sin(global_steering)])
        gg2 = - vf * np.array([np.cos(global_steering), np.sin(global_steering)])
        
        # Barrier and its partial derivatives
        h, g, H, K, time_vec = self.getBarrierAndPartials()

        # Lie derivatives
        start_time = time.time()
        h0 = float(h)
        Lfh0 = g @ xyd
        h1 = Lfh0 + self.alpha0 * h0

        Lfh1 = xyd.T @ H @ xyd + g @ f2 + self.alpha0 * Lfh0
        Lgh1 = g @ g2
        h2 = Lfh1 + np.abs(Lgh1) * umin_ + self.alpha1 * h1

        Lfh2 = 3 * xyd.T @ H @ f2 + np.dot(np.dot(np.dot(xyd, K), xyd), xyd) + (self.alpha0+self.alpha1) * (xyd.T @ H @ xyd + g @ f2) \
            + self.alpha0 * self.alpha1 * g @ xyd + g @ ff2 + np.sign(g @ g2) * (xyd.T @ H @ g2 + g @ fg2) * umin_
        Lgh2 = 2 * xyd.T @ H @ g2 + (self.alpha0+self.alpha1) * g @ g2 + g @ gf2 + np.sign(g @ g2) * g @ gg2 * umin_


        # Safe control computation
        num = -(Lfh2 + self.alpha2 * h2)
        if np.abs(Lgh2) < bound_:
            usafe_ = float(num * np.sign(Lgh2) / bound_)
        else:
            usafe_ = float(num / Lgh2)

        # Saturate so that it is within constraints
        if np.abs(usafe_) > self.max_steering_rate:
            usafe_ = np.sign(usafe_) * self.max_steering_rate

        # Stop time
        end_time = time.time()

        # Lie derivatives for data output
        Lfh1_Lgh1usafe_ch1 = Lfh1 + Lgh1 * usafe_ + self.alpha1 * h1
        Lfh1_Lgh1unom_ch1 = Lfh1 + Lgh1 * self.unom + self.alpha1 * h1
        Lfh2_Lgh2usafe_ch2 = Lfh2 + Lgh2 * usafe_ + self.alpha2 * h2
        Lfh2_Lgh2unom_ch2 = Lfh2 + Lgh2 * self.unom + self.alpha2 * h2
        
        if self.observe_computation_time:
            time_vec += end_time - start_time
        return usafe_, h2, h1, h0, Lfh2_Lgh2usafe_ch2, Lfh2_Lgh2unom_ch2, Lfh1_Lgh1usafe_ch1, Lfh1_Lgh1unom_ch1, time_vec


    def getBarrierAndPartials(self):

        # This function computes the barrier, its Jacobian and Hessian
        gamma = self.svm_params["gamma"]
        b = self.svm_params["b"]
        dc = self.svm_params["dc"]
        sv = self.svm_params["sv"]
        dim_sv = dc.shape[1]
        scaler = self.svm_params["scaler"]
        time_vec = 0
        
        # Scaler data plus related additional partial derivative
        xy = scaler.transform([self.x[:2]])[0]
        C = np.array(1/np.sqrt(scaler.var_))

        start_time = time.time()
        Xi_X = sv - xy
        Xi_X_ip = Xi_X ** 2
        exponent = -gamma * np.sum(Xi_X_ip, axis=1)
        Hi = np.exp(exponent)
        Hi_dc = Hi * dc

        # Barrier function
        hs = dc @ Hi
        h = hs + b - self.beta

        # Gradient
        G = np.multiply(Xi_X.T, [Hi, Hi]) * dc
        g = 2 * gamma * np.sum(G, axis=1) * C

        # Hessian
        H_non_scale = 2 * gamma * (2 * gamma * Xi_X.T * Hi_dc @ Xi_X - hs * np.eye(2))
        C_outer = np.outer(C, C)
        H = H_non_scale * C_outer

        # Third derivative (tensor)
        cross_terms = Xi_X[:, 0] * Xi_X[:, 1]
        V2 = np.array([2 * gamma * Xi_X[:, 0] ** 2 - 3, 2 * gamma * cross_terms, 2 * gamma * cross_terms,
                       2 * gamma * Xi_X[:, 1] ** 2 - 3]).T
        temp = Xi_X.T * Hi_dc @ V2
        K_non_scaled = 4 * gamma ** 2 * temp.reshape((2, 2, 2))
        C2_outer = (np.array([C_outer.ravel(), C_outer.ravel()]).T * C).reshape(2, 2, 2)
        K = K_non_scaled * C2_outer

        time_vec = (time.time() - start_time)
        return h, g, H, K, time_vec


def main(args=None):
    cbf_sf = CbfSafetyFilter()
    rate = rospy.Rate(1/cbf_sf.Ts)
    while not rospy.is_shutdown():
        cbf_sf.filter()
        rate.sleep()


if __name__ == '__main__':
    main()
