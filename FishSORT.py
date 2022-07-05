import numpy as np
from ReID import ResNet50
from Kalman import kalman_filter
from Target import target

class tracker(object):
    def __init__(self, reid = None, kalman = None):
        #Initialise ReID
        self.reid = ResNet50() if reid == None else reid
        
        #Initialise Kalman
        dt = 0.2
        self.kalman = kalman_filter(dt) if kalman == None else kalman

        #Initialise targets
        self.targs = []
        self.age_max = 10
        self.max_features = 100
    
    def update(self, detections, image):

        #Kalman predictions
        for t in self.targs:
            self.kalman.update_pred(t)
        
        #Associate
        cost_matrix_1, detection_features = self.reid.get_cost_matrix(self.targs, image, detections)
        np.append(detections, detection_features, axis = 1)

        matched, un_m_det, un_m_tar = self.associate(self.targs, detections, cost_matrix_1)

        keep = []
        for t in un_m_tar:
            self.kalman.update_state_no_detect(t)
            keep.append(t.age <= self.age_max)
        self.targs = un_m_tar[keep]

        for d in un_m_det:
            self.targs.append(target(self.kalman, d[:4], d[-1]))
        
        for m in matched:
            self.kalman.update_state(m[0], m[1])
            m[0].features.append(m[1])
            l = len(m[0].features)
            if(l > self.max_features):
                m[0].features = m[0].features[(l - self.max_features):]
    
    def associate(self, targets, detections, cost_matrix):
        all_targs = []
        all_ass = []
        for n in range(1, self.age_max):
            ts = [t.age == n for t in targets]
            ass = np.argmin(cost_matrix[ts], axis = 1)
            ass.flatten()
            cost_matrix = np.delete(np.delete(cost_matrix, ts, axis = 0), ass, axis = 1)
            np.append(all_targs, targets, ts)
            np.delete(targets, ts)
            np.append(all_ass, detections, ass)
            np.delete(detections, ass)
            return np.array([all_targs, all_ass]).T, detections, targets
            

