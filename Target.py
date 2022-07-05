import numpy as np

class target(object):
    def __init__(self, kalman, bbox, features): #bbox -> [x, y, w, h, ...]
        #Initialize target
        self.age = 1

        #Initialize ReID
        self.features = [features]

        #Initialize Kalman
        self.in_size = 7

        self.kalman = kalman
        self.state = np.append(kalman.conv_param(bbox[:4], [[0], [0], [0]], axis = 0))#[[x], [y], [h], [w/h], [x'], [y'], [h']] x.y -> center
        self.pred_state = self.state
        
        self.cov = kalman.get_init_cov(self.state[:4])
        self.pred_cov = self.cov