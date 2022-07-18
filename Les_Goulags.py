#Dependencies
import numpy as np

from Le_detroit_de_Bab_el_Mandeb_le_communiste import Aquagym

#Script pour entrainer des poids de base pour Nileman

theta = 0
dt = 0.1
w, h = 600, 600
r = 200
c_x, c_y = 2, 0.5 #Coeff de changement des vitesses
rs_list = []
ds_list = []
dt_list = []
missed = []
for f in range(1000):
    d_theta = (2*np.pi/20 * dt) % (2 * np.pi)
    theta += d_theta
    rs_list.append([np.array([r * np.sin(theta * c_x) + w/2, r * np.cos(theta * c_y) + h/2, 20 * (1 + np.sin(theta) / 10), 40 * (1 + np.sin(theta) / 10), 
    r * c_x * d_theta / dt * np.cos(theta), -r * c_y * d_theta / dt * np.sin(theta), 20 * np.cos(theta) / 10 * d_theta / dt, 40 * np.cos(theta) / 10 * d_theta / dt])])
    ds_list.append([rs_list[-1][0][:4] + (np.random.rand(4) - 0.5) * r * 0.05])
    dt_list.append(dt)
    missed.append([1 if f > 500 and np.random.rand() < 0.5 else 0])

gym = Aquagym(None)
gym.model = gym.create_Keras([[3, 10], [10, 8]]) #([[3, 10], [100, 100], [100, 50], [50, 8]])
gym.run(w, h, ds_list, rs_list, dt_list, missed_detections = missed, display = False)
print("Run Done")
gym.save("aqua_weights_test")