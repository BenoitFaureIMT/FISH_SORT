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
    
    def extract_sub_image_v2(self, img, bbox):#bbox -> [x, y, w, h]
        h, w, c = img.shape
        return img[bbox[1]*h:(bbox[1] + bbox[3])*h,bbox[0]*w:(bbox[0] + bbox[2])*w]

    def extract_features(self, img):
        img = cv2.resize(img, self.input_shape, cv2.INTER_LINEAR)/255
        img = img / 255
        f = self.model.predict(img)
        return f / np.linalg.norm(f)
    
    def get_distance(self, t1, t2, dist = "EuclideanDistance"):
        return self.distances[dist](t1, t2)
    
    def get_features(self, img, bbox):
        return self.extract_features(self.extract_sub_image_v2(img, bbox))
    
    def extract_cost_matrix(self, targets, detection_features):
        cost_matrix = np.zeros((len(targets), len(detection_features)))
        for i in range(cost_matrix.shape[0]):
            for j in range(cost_matrix.shape[1]):
                cost_matrix[i, j] = min([self.get_distance(tFeature, detection_features[j]) for tFeature in targets[i].features]) #d(i, j) = min{dist(rj, ri_k) | ri_k in Ri}
        return cost_matrix
    
    def get_cost_matrix(self, targets, img, bboxs):
        detection_features = [self.get_features(img, bbox) for bbox in bboxs]
        return self.extract_cost_matrix(targets, detection_features), detection_features
    
    def extract_associations(self, cost_matrix):
        return np.argmin(cost_matrix, axis = 1)

    def euclidean_distance(self, t1, t2):
        return np.linalg.norm(t1 - t2)
    
    distances = {
        "EuclideanDistance" : euclidean_distance
    }