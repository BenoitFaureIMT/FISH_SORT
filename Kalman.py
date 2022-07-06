import numpy as np
import scipy.linalg as lalg

import utils

#Note : input (x1, y1, x2, y2)

class kalman_filter(object):
    def __init__(self, dt):
        #Init params
        self.in_size = 7

        self.dt = dt
        self.update_matrix = self.get_update_matrix(dt)

        self.observation_matrix = np.eye(self.in_size, self.in_size) #maybe just remove cause we dont need all the observation_matrix, or maybe adapt to fisheye?

    #------------------------------------------------------Init------------------------------------------------------
    def get_update_matrix(self, dt):
        return np.array(
            [[1,0,0,0,dt,0,0],
            [0,1,0,0,0,dt,0],
            [0,0,1,0,0,0,dt],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]], dtype=np.float32)
    
    def conv_param(self, param):
        return np.array([[param[0] + param[2]/2], [param[1] + param[3]/2], [param[3]], [param[2]/param[3]]])
    
    def get_init_cov(self, param):
        std = np.array([.1, .1, .1, .1, 2, 2, 2])
        return np.diag(np.square(std)) 
    
    #------------------------------------------------------Util------------------------------------------------------
    def center_to_corner(self, v):
        w = v[2] * v[3]
        h = v[3]
        return [v[1] - w/2, v[0] - h/2, w, h]

    #-----------------------------------------------------Update-----------------------------------------------------
    def update_pred(self, targ):
        targ.pred_stzate = np.matmul(self.update_matrix, targ.state) #No controled input
        targ.pred_cov = np.matmul(np.matmul(self.update_matrix, targ.cov), self.update_matrix.T) #+Q (ignored)
    
    def update_state_no_detect(self, targ):
        targ.state = targ.pred_state
        targ.cov = targ.pred_cov
        targ.age += 1

    def update_state(self, targ, detect):
        R_cov_matrix = self.get_init_cov(targ.pred_state)
        R_cov_matrix *= (1 - detect[4])
        innovation_cov = np.matmul(np.matmul(self.observation_matrix, targ.pred_cov), self.observation_matrix.T) + R_cov_matrix

        #Going through Choleski decomposition to invert matrix (need to understand this better) - Maybe dont need all the transpose if we take upper
        cho_factor, lower = lalg.cho_factor(innovation_cov, lower = True, check_finite = False)
        kalman_gain = lalg.cho_solve((cho_factor, lower), np.matmul(targ.pred_cov, self.observation_matrix.T).T, check_finite=True).T

        param = utils.yxyx_kalman(detect[:4])
        param = np.append(param, (param[:3] - targ.state[:3])/self.dt, axis = 0)
        innovation_mean = param - np.matmul(self.observation_matrix, targ.pred_state)

        #Update
        targ.state = targ.pred_state + np.matmul(kalman_gain, innovation_mean)
        targ.cov = np.matmul(np.eye((self.in_size)) - np.matmul(kalman_gain, self.observation_matrix), targ.pred_cov)