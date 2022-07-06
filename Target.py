from cv2 import KalmanFilter
import numpy as np
import utils

class target(object):
    def __init__(self, kalman, bbox, features): #bbox -> [x, y, w, h, ...]
        #Initialize target
        self.age = 1

        #Initialize ReID
        self.features = [features]

        #Initialize Kalman
        self.kalman = kalman
        self.state = np.append(utils.yxyx_kalman(bbox[:4]), [[0], [0], [0]], axis = 0)#[[y], [x], [h], [w/h], [y'], [x'], [h']] y,x -> center
        self.pred_state = self.state
        
        self.cov = self.kalman.get_init_cov(self.state[:4])
        self.pred_cov = self.cov