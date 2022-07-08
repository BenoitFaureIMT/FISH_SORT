import numpy as np
import cv2

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
    def __init__(self, reid = None, kalman = None):
        #Initialise ReID
        self.reid = ResNet50() if reid == None else reid
        
        #Initialise Kalman
        dt = 0.2
        self.kalman = kalman_filter(dt) if kalman == None else kalman

        #Initialise targets
        self.targs = np.array([])
        self.age_max = 2
        self.max_features = 100

        #Initialise association
        self.lmbd = 0.5
    
    def update(self, detections, image):

        #Kalman predictions
        for t in self.targs:
            self.kalman.update_pred(t)
        
        #Run ReID
        #   Get cost matrix and store features
        cost_matrix_1, detection_features = self.reid.get_cost_matrix(self.targs, image, detections)
        print(cost_matrix_1)
        detections = np.append(detections, detection_features, axis = 1)

        #Run IoU
        cost_matrix_2 = np.array([[1 - utils.IoU_n(dtc, utils.kalman_xyxy(tar.pred_state)) for dtc in detections] for tar in self.targs]) #TODO Need more efficient
        print(cost_matrix_2)

        #Final cost matrix
        cost_matrix = cost_matrix_1
        if not (len(self.targs) == 0 or len(detections) == 0):
            cost_matrix = self.lmbd * cost_matrix_1 + (1 - self.lmbd) * cost_matrix_2

        #Associate
        m_det, m_tar, un_m_det, un_m_tar = self.associate(self.targs, detections, cost_matrix)
        print(len(m_det), len(m_tar), len(un_m_det), len(un_m_tar))
        print(m_tar.shape)
        #   Process targets with no associated detection
        keep = []
        for t in un_m_tar:
            self.kalman.update_state_no_detect(t)
            keep.append(t.age <= self.age_max)
        #if len(keep) != 0:
        self.targs = un_m_tar[keep]
        
        #   Process detections with no associated target
        t = list(self.targs) #TODO FIX THIS SHIT
        for d in un_m_det:
            t.append(target(self.kalman, d[:4], d[6:]))
            #np.append(self.targs, target(self.kalman, d[:4], d[-1]))
        self.targs = np.array(t)
        
        #   Process associated detections and targets
        for i in range(len(m_det)):
            self.kalman.update_state(m_tar[i], m_det[i])
            m_tar[i].features.append(m_det[i][6:])
            l = len(m_tar[i].features)
            if(l > self.max_features):
                m_tar[i].features = m_tar[i].features[(l - self.max_features):]
    
    def associate(self, targets, detections, cost_matrix):
        all_targs = np.zeros((0,))
        all_ass = np.zeros((0,2054))
        for n in range(1, self.age_max):
            ts = [t.age == n for t in targets]
            ic = cost_matrix[ts]
            if(len(ic) == 0):
                continue
            ass = np.argmin(ic, axis = 1)
            print(ass)
            cost_matrix = np.delete(np.delete(cost_matrix, ts, axis = 0), ass, axis = 1)
            #all_targs += targets[ts]
            all_targs = np.append(all_targs, targets[ts], axis = 0)
            targets = np.delete(targets, ts, axis = 0)
            #all_ass += [d for d in detections[ass]]
            all_ass = np.append(all_ass, detections[ass], axis = 0)
            detections = np.delete(detections, ass, axis = 0)
        return all_ass, all_targs, detections, targets
            

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

    print(len(tr.targs))
    # display(tr.targs, frame)
    # display_yolo(out, frame)
    #display_both(tr.targs, out, frame)
    #cv2.waitKey(1)

cam.release()
cv2.destroyAllWindows()