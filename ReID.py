import tensorflow as tf
import numpy as np
import cv2

class ResNet50(object):
    def __init__(self, weights = "imagenet", input_shape = (224, 224), pooling = "avg"):
        self.model = tf.keras.applications.ResNet50(
            include_top=False,
            weights=weights,
            input_shape=input_shape + (3,),
            pooling=pooling)
        self.input_shape = input_shape
    
    def extract_sub_image(self, img, bbox): #bbox -> [x1,y1,x2,y2] (x1, y1) top left, (x2, y2) botto right
        h, w, c = img.shape
        return img[bbox[1]*h:bbox[3]*h, bbox[0]*w:bbox[0]*w]

    def extract_features(self, img):
        img = cv2.resize(img, self.input_shape, cv2.INTER_LINEAR)/255
        img = img / 255
        return self.model.predict(img)
    
    def get_distance(self, t1, t2, dist = "EuclideanDistance"):
        return self.distances[dist](t1, t2)
    
    def get_features(self, img, bbox):
        return self.extract_features(self.extract_sub_image(img, bbox))
    
    def euclidean_distance(self, t1, t2):
        return np.linalg.norm(t1 - t2)
    
    distances = {
        "EuclideanDistance" : euclidean_distance
    }