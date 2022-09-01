#!/usr/bin/env python

import rospy
from ukf import UKF
import numpy as np
from numpy.linalg import inv
import math 
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import Point

# time variables
time_last = 0
dt = 0
time_now = 0

# pre-declare variables
state_dim = 59 # [x1 v1 a1 w1 dw1 E1 x2 v2 a2 w2 dw2 E2 xp vp ap wp dwp F1 F2]
measurement_dim =  18 # [x1 w1 tau1 x2 w2 tau2]
sensor_data = np.zeros(measurement_dim)
RotMat_ned = np.zeros((3,3))
RotMat_enu = np.zeros((3,3))

AlloMat = np.array([
                     [1,   1,   1,   1],
                     [-0.1625, 0.1625, 0.1625, -0.1625],
                     [0.1625, 0.1625, -0.1625, -0.1625],
                     [-2e-2,   2e-2,   -2e-2,   2e-2]    
                   ])
f_cmd_vec1 = np.zeros(4)
f_cmd_vec2 = np.zeros(4)
f_M1 = np.zeros(4)
f_M2 = np.zeros(4)
acc_enu1 = np.zeros(3)
acc_enu2 = np.zeros(3)
acc_enu_dyn1 = np.zeros(3)
acc_enu_dyn2 = np.zeros(3)
W1 = np.zeros(3)
W2 = np.zeros(3)
e3 = np.array([0,0,1])
f1_cmd1 = 0
f2_cmd1 = 0
f3_cmd1 = 0
f4_cmd1 = 0
f1_cmd2 = 0
f2_cmd2 = 0
f3_cmd2 = 0
f4_cmd2 = 0
acc_dyn1 = np.zeros(3)
acc_dyn2 = np.zeros(3)

debug = np.zeros(3)

estimate_state_list = Float32MultiArray()
acc_dyn1_list = Float32MultiArray()
time_now_list = Float64MultiArray()
debug_list = Float32MultiArray()


# Process Noise
q = np.eye(state_dim)
# quadrotor 1
# x,v,a
q[0][0] = 0.001 
q[1][1] = 0.001
q[2][2] = 0.001
q[3][3] = 0.001
q[4][4] = 0.001
q[5][5] = 0.001
q[6][6] = 0.001
q[7][7] = 0.001
q[8][8] = 0.001
# W,dW
q[9][9] = 0.001
q[10][10] = 0.001
q[11][11] = 0.001
q[12][12] = 0.001
q[13][13] = 0.001
q[14][14] = 0.001
# E
q[15][15] = 0.00001
q[16][16] = 0.00001
q[17][17] = 0.00001
q[18][18] = 0.00001

# quadrotor 2
# x,v,a
q[19][19] = 0.001 
q[20][20] = 0.001
q[21][21] = 0.001
q[22][22] = 0.001
q[23][23] = 0.001
q[24][24] = 0.001
q[25][25] = 0.001
q[26][26] = 0.001
q[27][27] = 0.001
# W,dW
q[28][28] = 0.001
q[29][29] = 0.001
q[30][30] = 0.001
q[31][31] = 0.001
q[32][32] = 0.001
q[33][33] = 0.001
# E
q[34][34] = 0.00001
q[35][35] = 0.00001
q[36][36] = 0.00001
q[37][37] = 0.00001

# payload
# x,v,a
q[38][38] = 0.001 
q[39][39] = 0.001
q[40][40] = 0.001
q[41][41] = 0.001
q[42][42] = 0.001
q[43][43] = 0.001
q[44][44] = 0.001
q[45][45] = 0.001
q[46][46] = 0.001
# W,dW
q[47][47] = 0.001
q[48][48] = 0.001
q[49][49] = 0.001
q[50][50] = 0.001
q[51][51] = 0.001
q[52][52] = 0.001
# F1,F2
q[53][53] = 0.00001
q[54][54] = 0.00001
q[55][55] = 0.00001
q[56][56] = 0.00001
q[57][57] = 0.00001
q[58][58] = 0.00001

# create measurement noise covariance matrices
p_yy_noise = np.eye(measurement_dim)
p_yy_noise[0][0] = 0.001
p_yy_noise[1][1] = 0.001
p_yy_noise[2][2] = 0.001
p_yy_noise[3][3] = 0.001
p_yy_noise[4][4] = 0.001
p_yy_noise[5][5] = 0.001
p_yy_noise[6][6] = 0.001
p_yy_noise[7][7] = 0.001
p_yy_noise[8][8] = 0.001
p_yy_noise[9][9] = 0.001
p_yy_noise[10][10] = 0.001
p_yy_noise[11][11] = 0.001
p_yy_noise[12][12] = 0.001
p_yy_noise[13][13] = 0.001
p_yy_noise[14][14] = 0.001
p_yy_noise[15][15] = 0.001
p_yy_noise[16][16] = 0.001
p_yy_noise[17][17] = 0.001

# create initial state
initial_state = np.zeros(state_dim)


def iterate_x(x, timestep):
    '''this function is based on the x_dot and can be nonlinear as needed'''
    global acc_enu, acc_enu_dyn, f_M_cmd1, f_M_cmd2, e3, debug
    global f1_cmd1, f2_cmd1, f3_cmd1, f4_cmd1, f1_cmd2, f2_cmd2, f3_cmd2, f4_cmd2
    global f_cmd_vec1, RotMat_enu, AlloMat, acc_dyn1
    global F1_state, F2_state
    m1 = 1.1
    m2 = 1.1
    mp = 0.16
    ms = 2.36
    J1 = np.diag([0.0117,0.0117,0.0222])
    J2 = np.diag([0.0117,0.0117,0.0222])
    Jp = np.diag([0.0088,0,0.0088])
    Js = np.diag([0.3952,0.0234,0.4105])
    g = 9.806
    E1_vec_from_x = x[15:19]
    E2_vec_from_x = x[34:38]
    E1_diag_from_x = np.diag(E1_vec_from_x)
    E2_diag_from_x = np.diag(E2_vec_from_x)
    f_cmd_vec1 = np.array([f1_cmd1,f2_cmd1,f3_cmd1,f4_cmd1])
    f_cmd_vec2 = np.array([f1_cmd2,f2_cmd2,f3_cmd2,f4_cmd2])
    f_M_cmd1 = np.dot(AlloMat,f_cmd_vec1)
    f_M_cmd2 = np.dot(AlloMat,f_cmd_vec2)

    W1_from_x = x[9:12]
    W2_from_x = x[28:31]

    # [f M] = AlloMat*E_diag*f_cmd_vec
    f_M_E1 = np.dot(AlloMat,np.dot(E1_diag_from_x,f_cmd_vec1))
    f_M_E2 = np.dot(AlloMat,np.dot(E2_diag_from_x,f_cmd_vec2))

    # F1, F2
    F1_state = x[53:56]
    F2_state = x[56:59]

    # Mp
    Mat1 = np.array([
                    [0,   0,   0],
                    [0,   0, 0.5],
                    [0,-0.5,   0]  
                    ])
    Mat2 = np.array([
                    [0,   0,   0],
                    [0,   0,-0.5],
                    [0, 0.5,   0]  
                    ])
    Mp = np.dot(Mat1,-F1_inertial) + np.dot(Mat2,-F2_inertial) 

    # dynamics
    # a = fRe3/m-ge3
    acc_dyn1 = f_M_cmd1[0]*np.dot(RotMat_enu,e3)/m1 - g*e3 + F1_sensor/m1
    acc_dyn2 = f_M_cmd2[0]*np.dot(RotMat_enu,e3)/m2 - g*e3 + F2_sensor/m2

    a_state1 = f_M_E1[0]*np.dot(RotMat_enu,e3)/m1 - g*e3 + F1_state/m1
    a_state2 = f_M_E2[0]*np.dot(RotMat_enu,e3)/m2 - g*e3 + F2_state/m2
    a_statep = -(F1_state+F2_state)/mp - g*e3
    
    # WxJW
    W1xJ1W1 = np.cross(W1_from_x,np.dot(J1,W1_from_x))
    W2xJ2W2 = np.cross(W2_from_x,np.dot(J2,W2_from_x))
    # M-WxJW
    M1_W1xJ1W1 = f_M_E1[1:4] - W1xJ1W1
    M2_W2xJ2W2 = f_M_E2[1:4] - W2xJ2W2
    # dW = inv(J)(M-WxJW)    
    dW1 = np.dot(inv(J1),M1_W1xJ1W1)
    dW2 = np.dot(inv(J2),M2_W2xJ2W2)
    # dWp_state
    dW1_state = x[12:15]
    dW2_state = x[31:34]
    dWp_state = np.dot(inv(Jp),Mp-(tau1_sensor+tau2_sensor))
    # tau_state tau = JdW - (M - WxJW) 
    tau_state1 = np.dot(J1,dW1) - (f_M_E1[1:4]-W1xJ1W1)
    tau_state2 = np.dot(J2,dW2) - (f_M_E2[1:4]-W2xJ2W2)


    ret = np.zeros(len(x))
    # quadrotor 1
    # x
    ret[0] = x[0] + x[3] * timestep + 0.5*x[6]*timestep*timestep
    ret[1] = x[1] + x[4] * timestep + 0.5*x[7]*timestep*timestep
    ret[2] = x[2] + x[5] * timestep + 0.5*x[8]*timestep*timestep
    # v
    ret[3] = x[3] + x[6] * timestep
    ret[4] = x[4] + x[7] * timestep
    ret[5] = x[5] + x[8] * timestep
    # a
    ret[6] = a_state1[0]
    ret[7] = a_state1[1]
    ret[8] = a_state1[2]
    # W			
    ret[9] = x[9] + x[12] * timestep
    ret[10] = x[10] + x[13] * timestep
    ret[11] = x[11] + x[14] * timestep
    # dW
    ret[12] = x[12]
    ret[13] = x[13]
    ret[14] = x[14]
    # E
    ret[15] = x[15]
    ret[16] = x[16]
    ret[17] = x[17]
    ret[18] = x[18]

    # quadrotor 2
    # x
    ret[19] = x[19] + x[22] * timestep + 0.5*x[25]*timestep*timestep
    ret[20] = x[20] + x[23] * timestep + 0.5*x[26]*timestep*timestep
    ret[21] = x[21] + x[24] * timestep + 0.5*x[27]*timestep*timestep
    # v
    ret[22] = x[22] + x[25] * timestep
    ret[23] = x[23] + x[26] * timestep
    ret[24] = x[24] + x[27] * timestep
    # a
    ret[25] = a_state2[0]
    ret[26] = a_state2[1]
    ret[27] = a_state2[2]
    # W			
    ret[28] = x[28] + x[31] * timestep
    ret[29] = x[29] + x[32] * timestep
    ret[30] = x[30] + x[33] * timestep
    # dW
    ret[31] = x[31]
    ret[32] = x[32]
    ret[33] = x[33]
    # E
    ret[34] = x[34]
    ret[35] = x[35]
    ret[36] = x[36]
    ret[37] = x[37]

    # payload
    # x
    ret[38] = x[38] + x[41] * timestep + 0.5*x[44]*timestep*timestep
    ret[39] = x[39] + x[42] * timestep + 0.5*x[45]*timestep*timestep
    ret[40] = x[40] + x[43] * timestep + 0.5*x[46]*timestep*timestep
    # v
    ret[41] = x[41] + x[25] * timestep
    ret[42] = x[42] + x[26] * timestep
    ret[43] = x[43] + x[27] * timestep
    # a
    ret[44] = a_statep[0]
    ret[45] = a_statep[1]
    ret[46] = a_statep[2]
    # W			
    ret[47] = x[47] + x[50] * timestep
    ret[48] = x[48] + x[51] * timestep
    ret[49] = x[49] + x[52] * timestep
    # dW
    ret[50] = dWp_state[0]
    ret[51] = dWp_state[1]
    ret[52] = dWp_state[2]

    # F1
    ret[53] = x[53]
    ret[54] = x[54]
    ret[55] = x[55]
    # F2
    ret[56] = x[56]
    ret[57] = x[57]
    ret[58] = x[58]
    return ret

def measurement_model(x):
    """
    :param x: states
    """
    # dynamics

    global measurement_dim
    ret = np.zeros(measurement_dim)
    # quadrotor 1
    # x
    ret[0] = x[0]
    ret[1] = x[1]
    ret[2] = x[2]
    # W
    ret[3] = x[9]
    ret[4] = x[10]
    ret[5] = x[11]
    # tau1
    ret[6] = tau_state1[0]
    ret[7] = tau_state1[1]
    ret[8] = tau_state2[2]
    # quadrotor 2
    ret[9] = x[19]
    ret[10] = x[20]
    ret[11] = x[21]
    # W
    ret[12] = x[28]
    ret[13] = x[29]
    ret[14] = x[30]
    # tau2
    ret[15] = tau_state2[0]
    ret[16] = tau_state2[1]
    ret[17] = tau_state2[2]
    # payload
    # x
    ret[18] = x[38]
    ret[19] = x[39]
    ret[20] = x[40]
    # W
    ret[21] = x[47]
    ret[22] = x[48]
    ret[23] = x[49]
    
    return ret

def pos_enu_cb(data):
    global pos_enu, sensor_data 
    pos_enu = np.array([data.x, data.y, data.z])
    sensor_data[0:3] = np.array([data.x, data.y, data.z])

def gyro_cb(data):
    global W, sensor_data
    W = np.array([data.x, data.y, data.z])
    sensor_data[3:6] = np.array([data.x, data.y, data.z])

def f1_cmd1_cb(data):
    global f1_cmd1
    f1_cmd1 = data.data[0]

def f2_cmd1_cb(data):
    global f2_cmd1
    f2_cmd1 = data.data[0]

def f3_cmd1_cb(data):
    global f3_cmd1
    f3_cmd1 = data.data[0]

def f4_cmd1_cb(data):
    global f4_cmd1
    f4_cmd1 = data.data[0]

def f1_cmd2_cb(data):
    global f1_cmd2
    f1_cmd1 = data.data[0]

def f2_cmd2_cb(data):
    global f2_cmd2
    f2_cmd1 = data.data[0]

def f3_cmd3_cb(data):
    global f3_cmd2
    f3_cmd1 = data.data[0]

def f4_cmd4_cb(data):
    global f4_cmd2
    f4_cmd1 = data.data[0]

def RotMat_ned_cb(data):
    global RotMat_ned, RotMat_enu, RotFrame
    RotMat_ned = np.array([
				[data.data[0],data.data[1],data.data[2]],
				[data.data[3],data.data[4],data.data[5]],
				[data.data[6],data.data[7],data.data[8]]
	       		  ])
    RotFrame = np.array([
			[0,1,0],
			[1,0,0],
			[0,0,-1]
		        ])
    RotMat_enu = np.dot(RotFrame,np.dot(RotMat_ned,RotFrame))

def ukf():
    global time_last, time_now
    global dt
    time_now = rospy.get_time()
    dt = rospy.Time.now().to_sec() - time_last
    ukf_module.predict(dt)
    ukf_module.update(measurement_dim, sensor_data, p_yy_noise)
    time_last = rospy.Time.now().to_sec()

    # print('dt:')
    # print(dt)
    # print('rospy.Time.now().to_sec()')
    # print(rospy.Time.now().to_sec())


if _name_ == "_main_":
    try:
        rospy.init_node('UKF')
        state_pub = rospy.Publisher("/ukf_estimated_state", Float32MultiArray, queue_size=10)
        acc_dyn_pub = rospy.Publisher("/ukf_acc_dyn1", Float32MultiArray, queue_size=10)
        time_now_pub = rospy.Publisher("/time_now", Float64MultiArray, queue_size=10)
        debug_pub = rospy.Publisher("/ukf_debug", Float32MultiArray, queue_size=10)
        rospy.Subscriber("/pos_enu", Point, pos_enu_cb, queue_size=10)
        rospy.Subscriber("/angular_vel", Point, gyro_cb, queue_size=10)
        rospy.Subscriber("/f1_cmd1", Float32MultiArray, f1_cmd1_cb, queue_size=10)
        rospy.Subscriber("/f2_cmd1", Float32MultiArray, f2_cmd1_cb, queue_size=10)
        rospy.Subscriber("/f3_cmd1", Float32MultiArray, f3_cmd1_cb, queue_size=10)
        rospy.Subscriber("/f4_cmd1", Float32MultiArray, f4_cmd1_cb, queue_size=10)
        rospy.Subscriber("/f1_cmd2", Float32MultiArray, f1_cmd2_cb, queue_size=10)
        rospy.Subscriber("/f2_cmd2", Float32MultiArray, f2_cmd2_cb, queue_size=10)
        rospy.Subscriber("/f3_cmd2", Float32MultiArray, f3_cmd2_cb, queue_size=10)
        rospy.Subscriber("/f4_cmd2", Float32MultiArray, f4_cmd2_cb, queue_size=10)
        rospy.Subscriber("/RotMat_ned", Float32MultiArray, RotMat_ned_cb, queue_size=10)

        # pass all the parameters into the UKF!
        # number of state variables, process noise, initial state, initial coariance, three tuning paramters, and the iterate function
        #def __init__(self, num_states, process_noise, initial_state, initial_covar, alpha, k, beta, iterate_function, measurement_model):
        ukf_module = UKF(state_dim, q, initial_state, 0.001*np.eye(state_dim), 0.001, 0.0, 2.0, iterate_x, measurement_model)
        rate = rospy.Rate(40)
        print("start ukf model!")
        while not rospy.is_shutdown():         
            ukf()
            estimate_state = ukf_module.get_state()
            estimate_state_list.data = list(estimate_state)
            state_pub.publish(estimate_state_list)
            
            acc_dyn1_list.data = list(acc_dyn1)
            acc_dyn1_pub.publish(acc_dyn1_list)
            
            time_now_list.data = [time_now]
            time_now_pub.publish(time_now_list)
            
            debug_list.data = list(debug)
            debug_pub.publish(debug_list)

            rate.sleep()
    except rospy.ROSInterruptException:
        pass