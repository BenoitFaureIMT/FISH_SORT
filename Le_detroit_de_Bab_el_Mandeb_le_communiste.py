#Dependencies
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import cv2

from Les_Communistes_Egyptiens import Nileman, Atlante

#Class to train the CNN of the Aquaman filter
class Aquagym(object):
    def __init__(self, keras_model):
        self.model = keras_model

    #Function to run the training
    #       - inputs : detections, realPositions, dts (detections need to correspond (same index) through out the frames)
    #       detections -> [ [[bbox], bboxes], frames ]
    #       realPositions -> same
    #       dts -> [dt_frame1-2, dt_frame2-3 ...]
    def run(self, width, height, detections, realPositions, dts, missed_detections = 0, display = False):
        if(min(len(detections), len(realPositions), len(dts)) < 2):
            print("I can't do shit to train if there is less than 2 frames you fucking retard!!!!!")
            print("Go fuck yourself")
            return
        
        AFilter = Nileman(width, height, "")
        AFilter.coeff_model = self.model
        targs = [Atlante(d) for d in detections[0]]

        if missed_detections == 0:
            missed_detections = np.zeros((len(detections), len(detections[0])))
    
        for i in range(1, len(detections)):

            inps = []
            ideals = []

            #Displaying results ish
            if display:
                img = np.zeros((width, height, 3))

            for j in range(len(targs)):

                AFilter.pred_next_state(targs[j], dts[i])
                if missed_detections[i][j] == 0:
                    err, inp = AFilter.update_state(targs[j], detections[i][j], dts[i])
                    inps.append(inp[0])
                    ideal = np.divide(np.array([realPositions[i][j]]).T - targs[j].pred_state, err, out = np.zeros_like(err), where=err!=0)
                    ideals.append(ideal.T[0])
                else:
                    AFilter.update_state_no_detection(targs[j], dts[i])

                #Displaying
                if display:
                    self.draw_rect(img, realPositions[i][j], (255, 255, 255))
                    self.draw_rect(img, targs[j].state.T[0], (0, 255, 0) if missed_detections[i][j] == 0 else (0, 0, 255))

            if len(inps) > 0:
                self.model.fit(np.array(inps), np.array(ideals), epochs=1, batch_size=1)

            if display:
                cv2.putText(img, str(i), (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("Le detroit de Bab-el-Mandeb", img)
                k = cv2.waitKey(1)
                if k == 27:
                    exit()
            
            print(i, "/", len(detections))
    
    #Function to save model
    def save(self, save_path):
        weight = []
        bias = []
        for l in self.model.layers:
            a = l.get_weights()
            weight.append(a[0])
            bias.append(a[1])
        np.save(save_path, [weight, bias], allow_pickle = True)
    
    #Function to create Keras Model
    #   layers -> [[inp size, out size], layer2...]
    def create_Keras(self, layers, optimizer = 'sgd', loss = tf.keras.losses.MeanSquaredError()):
        k_model = Sequential()

        for l in layers:
            k_model.add(Dense(int(l[1]), input_shape = (int(l[0]),), activation='sigmoid'))
        
        k_model.compile(optimizer = optimizer, loss = loss)
        return k_model

    #Utils for drawing
    def draw_rect(self, img, state, color):
        cv2.rectangle(img, (int(state[0] - state[2]/2), int(state[1] - state[3]/2)), (int(state[0] + state[2]/2), int(state[1] + state[3]/2)), color, 2)