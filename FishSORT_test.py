import numpy as np
import cv2
from scipy.spatial.distance import cdist
import lap

from ReID import ResNet50
from Kalman import kalman_filter
from Target import target
from Detection import YOLOv5
import utils

# Notes to self
# 1) bbox definitions:
#       Yolov5 -> [y_corner1, x_corner1, y_corner2, x_corner2]
#       Kalman <- [y_center, x_center, h, w/h]
#              -> [y_center, x_center, h, w/h]
#       Image extraction <- [y_corner1, x_corner1, y_corner2, x_corner2]
# 2) Flow:
#   Yolov5 -> extraction -> Kalman
#    Convert Flow:
#       Yolov5 (detection) -> Yolov5 (update) -> batch extraction -conv> Kalman

class tracker(object):
    def __init__(self, reid = None, kalman = None, association = None):
        #Initialise ReID
        self.reid = ResNet50() if reid == None else reid
        
        #Initialise Kalman
        dt = 0.2
        self.kalman = kalman_filter(dt) if kalman == None else kalman

        #Initialise targets
        self.targs = np.array([])
        self.age_max = 10
        self.max_features = 100

        #Initialise EMA
        self.alpha = 0.9
        self.former_feat = None


    
    def update(self, detections, image):

        IoU_threshold = 0.2
        cosine_threshold = 0.2
        cost_threshold = 0.2

        for t in self.targs:
            self.kalman.update_pred(t)
        
        #EMA
        
        
        cost_matrix_IoU, detection_features = self.reid.get_cost_matrix(self.targs, image, detections)
        detections = np.append(detections, detection_features, axis = 1)

        if (len(self.targs) == 0 or len(detections) == 0):
            cost_matrix = cost_matrix_IoU
        else:
            target_f = np.asarray([self.targs[i].features[0] for i in range(len(self.targs))])
            cost_matrix_cos = np.maximum(0.0, cdist(target_f, detection_features, metric='cosine')) / 2.0
            cost_matrix_IoU[cost_matrix_IoU > IoU_threshold] = 1.0
            cost_matrix_cos[cost_matrix_cos > cosine_threshold] = 1.0
            cost_matrix = np.minimum(cost_matrix_IoU, cost_matrix_cos)

        match, unm_tr, unm_det = self.associate(cost_matrix, cost_threshold)
        print(match, unm_tr, unm_det)

        for ind_track, ind_det in match:
            tracks = self.targs[ind_track]
            detect = detections[ind_det]
            self.kalman.update_state(tracks,detect)
            tracks.features.append(detect[6:])
            l = len(tracks.features)
            if l > self.max_features:
                tracks.features = tracks.features[(l - self.max_features):]

        new_targs = []
        for ind_unm_tr in unm_tr:
            unm_tracks = self.targs[ind_unm_tr]
            self.kalman.update_state_no_detect(unm_tracks)
            if unm_tracks.age <= self.age_max:
                new_targs.append(unm_tracks)
        
        for ind_unm_det in unm_det:
            unm_detect = detections[ind_unm_det]
            new_targs.append(target(self.kalman, unm_detect[:4], unm_detect[6:]))
        self.targs = np.array(new_targs)
    
    def associate(self, cost_mat, cost_thres):
        if cost_mat.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_mat.shape[0])), tuple(range(cost_mat.shape[1]))
        matches, unmatch_track, unmatch_detection = [], [], []
        __, x, y = lap.lapjv(cost_mat, extend_cost=True, cost_limit=cost_thres)
        for ix, mx in enumerate(x):
            if mx >= 0:
                matches.append([ix, mx])
        unmatch_track = np.where(x < 0)[0]
        unmatch_detection = np.where(y < 0)[0]
        matches = np.asarray(matches)
        return matches, unmatch_track, unmatch_detection
    
    def EMA(self, feat):
        if self.former_feat is None:
            self.former_feat = feat
            return feat
        else:
            new_feat = self.alpha * self.former_feat + (1 - self.alpha) * feat
            self.former_feat = new_feat
            return new_feat


def display(targs, img):
    for t in targs:
        y,x,h,wh = t.state[0]*640,t.state[1]*640,t.state[2]*640,t.state[3]
        w = wh*h
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (255,0,0), 1)

    cv2.imshow('BBox', img)

def display_yolo(out, img):
    for o in out:
        y,x,y2,x2 = o[:4]*640
        cv2.rectangle(img, (int(x), int(y)), (int(x2), int(y2)), (0,255,0), 1)
    
    cv2.imshow('Yolov5', img)

def display_both(targs, out, img):
    for o in out:
        y,x,y2,x2 = o[:4]*640
        cv2.rectangle(img, (int(x), int(y)), (int(x2), int(y2)), (0,255,0), 1)
    
    for t in targs:
        y,x,h,wh = t.state[0]*640,t.state[1]*640,t.state[2]*640,t.state[3]
        w = wh*h
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (255,0,0), 1)
    
    cv2.imshow('Yolov5 + FishSORT', img)

tr = tracker()
det = YOLOv5("saved_model")
det.warm_up()

cam = cv2.VideoCapture("testfish.avi")
ret = True
while ret:
    print("--------")
    ret, frame = cam.read()
    frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
    out = det.process_output(det.run_net(frame))
    print(len(out))

    tr.update(out, frame)

    #print(len(tr.targs))
    #if len(tr.targs) > 0: 
    #print(tr.targs[0].state)
    display(tr.targs, frame)
    #display_yolo(out, frame)
    display_both(tr.targs, out, frame)
    cv2.waitKey(1)

cam.release()
cv2.destroyAllWindows()