import numpy as np

def xywh_xyxy_b(batch):
    nbatch = np.zeros(batch.shape)
    nbatch[:,:2] = batch[:,:2]
    nbatch[:,2] = batch[:,0] + batch[:,2]
    nbatch[:,3] = batch[:,1] + batch[:,3]
    nbatch[:,4:] = batch[:,4:]
    return nbatch

def xywh_xyxy(state):
    return np.array([state[0], state[1], state[0] + state[2], state[1] + state[3]])

def kalman_xyxy(state):
    h = state[3][0] / 2
    w = state[4][0] * h
    return np.array([state[0][0] - h, state[1][0] - w, state[0][0] + h, state[1][0] + w])

def xyxy_kalman_b(batch):
    nbatch = np.zeros(batch.shape)
    nbatch[:,0] = (batch[:,0] + batch[:,2])/2
    nbatch[:,1] = (batch[:,1] + batch[:,3])/2
    nbatch[:,2] = np.abs(batch[:,3] - batch[:,1])
    nbatch[:,3] = np.abs(batch[:,2] - batch[:,0])/batch[:,2]
    return nbatch

def yxyx_kalman(state):
    h = np.abs(state[2] - state[0])
    return np.array([[(state[0] + state[2]/2)], [(state[1] + state[3]/2)], [h], [np.abs(state[3] - state[1])/h]])

def IoU_b(dtc, tar): #Needs to be revesited
            tar = tar.flatten()

            dt_l_y = dtc[0]
            dt_r_y = dtc[2]
            if(dtc[0] > dtc[2]):
                dt_l_y = dtc[2]
                dt_r_y = dtc[0]
            dt_l_x = dtc[1]
            dt_r_x = dtc[3]
            if(dtc[1] > dtc[3]):
                dt_l_x = dtc[3]
                dt_r_x = dtc[1]

            ly = max(dt_l_y, tar[0])
            lx = max(dt_l_x, tar[1])
            ry = min(dt_r_y, tar[0] + tar[2])
            rx = min(dt_r_x, tar[1] + tar[3] * tar[2])
            I = (rx - lx) * (ry - ly)
            if I <= 0:
                return 0
            return ((dt_r_y - dt_l_y)*(dt_r_x - dt_l_x) + tar[2] * tar[3] * tar[2]) / I - 1

def IoU_n(box1, box2) :
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    width = (x2 - x1)
    height = (y2 - y1)
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    area_combined = area1 + area2 - area_overlap
    iou = area_overlap / area_combined
    return iou