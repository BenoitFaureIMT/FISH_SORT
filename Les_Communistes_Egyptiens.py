#Dependences
import numpy as np
import cv2
import time

#Custom adaptation of the Kalman filter - which is not a Kalman filter at all
#       -> Replaces Kalman's inovation with a NN
#Requirements for targets
#   -> state = [yc, xc, h, w, x', y', ?]
class Nileman(object):
    def __init__(self, screen_w, screen_h, weight_path):
        #CNN params
        self.coeff_model = self.FCNN(weight_path)

        #Motion params
        self.screen_w = screen_w
        self.screen_h = screen_h

    #--------------------------NN functions--------------------------
    #   NN class
    #       -> The network is fully connected and small => nothing complicated
    #       -> Keras has to many things slowing down calculations for such a small network
    #       -> So we make a custom feedforward class
    #       -> Obligatory sigmoid because I am lazy
    class FCNN(object):
        def __init__(self, weight_path):
            if(weight_path == ""):
                return None
            all = np.load(weight_path, allow_pickle = True)
            self.weights = all[0]
            self.bias = all[1]
            self.n = len(self.weights)
        
        def sigmoid(self, inp):
            exp = np.exp(inp)
            return exp / (1 + exp)
        
        def predict(self, inp):
            for i in range(self.n):
                inp = np.matmul(inp, self.weights[i]) + self.bias[i]
                inp = self.sigmoid(inp)
            return inp

    #--------------------------State functions--------------------------
    #   State prediction
    def pred_next_state(self, targ, dt):
        targ.pred_state = targ.state
        targ.pred_state[:4] += targ.state[4:] * dt
    
    #   State update - no detection
    def update_state_no_detection(self, targ, dt):
        targ.state = targ.pred_state

        targ.missed_detection = True
        targ.time_since_last_detection += dt

    #   State update - detection (detect -> [xc, yc, w, h])
    def update_state(self, targ, detect, dt):
        #Calculate error vector
        #   Calculate detection
        dpx, dpy, dpw, dph, ndt = 0, 0, 0, 0, 0
        if(targ.missed_detection): # TODO Make sure this is fine - here dt could be big and fuck up the taylor dev
            dpx = detect[0] - targ.last_detected_state[0,0]
            dpy = detect[1] - targ.last_detected_state[1,0]
            dpw = detect[2] - targ.last_detected_state[2,0]
            dph = detect[2] - targ.last_detected_state[2,0]
            ndt = targ.time_since_last_detection
            targ.missed_detection = False
            targ.time_since_last_detection = 0
        else:
            dpx = detect[0] - targ.state[0,0]
            dpy = detect[1] - targ.state[1,0]
            dpw = detect[2] - targ.state[2,0]
            dph = detect[3] - targ.state[3,0]
            ndt = dt
        vx = dpx / ndt
        vy = dpy / ndt
        vw = dpw / ndt
        vh = dph / ndt
        detected_state = np.array([[detect[0], detect[1], detect[2], detect[3], vx, vy, vw, vh]]).T
    	#   Calculate error
        err = detected_state - targ.pred_state

        #Calculate interpolation vector
        inp = np.array([[err[0][0] / self.screen_w, err[1][0] / self.screen_h, 0 if targ.missed_detection else 1]])
        coeff = self.coeff_model.predict(inp)
        print(coeff)

        #Update state
        targ.state = targ.pred_state + coeff.T * err
        targ.last_detected_state = targ.state

        return err, inp

#Class to define the targets
class Atlante(object):
    def __init__(self, detection):
        #Init states
        self.state = np.array([[detection[0], detection[1], detection[2], detection[3], 0, 0, 0, 0]]).T
        self.pred_state = self.state

        #Init missed
        self.last_detected_state = self.state
        self.missed_detection = False
        self.time_since_last_detection = 0

f = 1
s = 600

def get_alpha(xr, yr): #xr, yr -> real world position at distance f
    d = np.sqrt(xr*xr + yr*yr)
    return np.abs(np.arctan(d/f))

def get_distorted_r(alpha):
    return f*np.sin(alpha)

def get_theta(xr, yr):
    t = np.arccos(xr/np.sqrt(xr*xr + yr*yr))
    return t if yr > 0 else 2*np.pi - t

def get_fish_box(xl, yl, xr, yr):
    a1 = get_alpha(xl, yl)
    a2 = get_alpha(xr, yr)
    d1 = get_distorted_r(a1)
    d2 = get_distorted_r(a2)
    t1 = get_theta(xl, yl)
    t2 = get_theta(xr, yr)
    
    nx1 = d1 * np.cos(t1) * s/2
    ny1 = d1 * np.sin(t1) * s/2
    nx2 = d2 * np.cos(t2) * s/2
    ny2 = d2 * np.sin(t2) * s/2

    return nx1 + s/2, ny1 + s/2, np.abs(nx1 - nx2), np.abs(ny1 - ny2)

if False:
    def draw_rect(img, state, color):
        cv2.rectangle(img, (int(state[0] - state[2]/2), int(state[1] - state[3]/2)), (int(state[0] + state[2]/2), int(state[1] + state[3]/2)), color, 2)
    theta = 0
    dt = 0.1
    real_state = np.array([0, 200, 10, 20])
    real_state = [*get_fish_box(real_state[0], real_state[1], real_state[0] + real_state[2], real_state[1] + real_state[3])]
    targ = Atlante(real_state)
    filt = Nileman(600, 600, "aqua_weights_test.npy")
    while True:
        t = time.perf_counter()
        theta += (2*np.pi/5 * dt) % (2 * np.pi)
        real_state = np.array([200 * np.sin(theta * 2) + 300, 200 * np.cos(theta / 2) + 300, 10, 20])
        #real_state = [*get_fish_box(real_state[0], real_state[1], real_state[0] + real_state[2], real_state[1] + real_state[3])]

        a = time.perf_counter()
        filt.pred_next_state(targ, 0.1)
        filt.update_state(targ, real_state, 0.1)

        print("Nileman speed : ", int((time.perf_counter() - a) * 100000)/100, " ms")
        print("--------------------")

        img = np.zeros((600, 600, 3), np.uint8)
        draw_rect(img, real_state, (255, 255, 255))
        draw_rect(img, targ.state.T[0], (0, 0, 255))
        cv2.imshow("Bob", img)
        k = cv2.waitKey(1)
        dt = time.perf_counter() - t
        if k == 27:
            break