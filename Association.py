from ReID import ResNet50
import utils

import numpy as np
from scipy.optimize import linear_sum_assignment


class association_algo(object):
    def __init__(self, lmbd, age_max):
        self.lmbd = lmbd
        self.age_max = age_max

        self.detection_features = detection_features
        self.mat_cov = mat_cov
        self.predictions = predictions



    def mahal_distance(self, i, j, detections, predictions, mat_cov):
        mat = detections[:][j]-predictions[i]
        return np.dot(np.dot(np.transpose(mat),mat_cov),mat)
    
    def smallest_cosine_distance(self, target, detection_features):
        return self.reid.extract_cost_matrix(target, detection_features)
    
    def mahal_conf(self, i, j, detections, predictions, mat_cov, target, detection_features):
        return np.dot(self.mahal_conf_1(i, j, detections, predictions, mat_cov),self.mahal_conf_2(target, detection_features))
    
    def mahal_conf_1(self, i, j, detections, predictions, mat_cov):
        if self.mahal_distance(i, j, detections, predictions, mat_cov) <= 9.4877:
            return 1
        else:
            return 0
    
    def mahal_conf_2(self, target, detection_features):
        if self.smallest_cosine_distance(target, detection_features) <= 5.9915:
            return 1
        else:
            return 0
    
    def weighted_sum(self, i, j, detections, predictions, mat_cov, target, detection_features):
        return self.lmbd*self.mahal_distance(i, j, detections, predictions, mat_cov)+(1-self.lmbd)*self.smallest_cosine_distance(target, detection_features)
    
    def dist_metric(self, weighted_sum, ind_i, ind_j, detections, predictions, mat_cov, target, detection_features, N, M):
        return [[weighted_sum(ind_i, ind_j, detections, predictions, mat_cov, target, detection_features) for i in range(N-1)] for j in range(M-1)]

    def hungarian(self, func_dist_metric, T, U, track_indices, max_dist=0.9):
        if len(U) == 0 or len(T) == 0:
            return [], T, U
        cost_matrix = func_dist_metric(self, self.weighted_sum, T, U)
        cost_matrix[cost_matrix > max_dist] = max_dist + 1e-5 # TODO Coeff a potentiellement changer
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matches, unmatched_tracks, unmatched_detections = [], [], []
        for col, detection_idx in enumerate(U):
            if col not in col_indices:
                unmatched_detections.append(detection_idx)
        for row, track_idx in enumerate(track_indices):
            if row not in row_indices:
                unmatched_tracks.append(track_idx)
        for row, col in zip(row_indices, col_indices):
            track_idx = track_indices[row]
            detection_idx = U[col]
            if cost_matrix[row, col] > max_dist:
                unmatched_tracks.append(track_idx)
                unmatched_detections.append(detection_idx)
            else:
                matches.append((track_idx, detection_idx))
        return matches, unmatched_tracks, unmatched_detections
    
    def match_cascade(self, target, detections, reid):
        N = len(target)
        M = len(detections)
        track_indices =  np.arange(1,N)
        detect_indices = np.arange(1,M)
        detections = utils.yxyx_kalman(detections)
        gate_matrix = [[self.mahal_conf(i,j) for i in range(N)] for j in range(M)]
        matches = []
        unmatched = detect_indices
        for i in range(1,self.age_max):
            Tn = []
            for j in range(N):
                if j>0: # TODO Changer condition : Tous les tracks non sélectionnés lors des n frames précédentes
                    Tn.append(track_indices[j])
            X = self.hungarian(self, self.dist_metric, target, detections, Tn, unmatched, max_dist=0.9)[0]
            for k in range(N):
                for l in range(M):
                    if gate_matrix[i][j]*X[i][j] > 0:
                        matches.append((k,l))
            for m in range(M): # Peut être optimisé en changeant l'odre des boucles
                if sum([gate_matrix[i][m]*X[i][m] for i in range(N)]):
                    unmatched.remove(m)
        return matches, unmatched