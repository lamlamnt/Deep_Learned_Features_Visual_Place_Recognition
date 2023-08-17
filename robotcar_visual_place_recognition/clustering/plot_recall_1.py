import numpy as np
import matplotlib.pyplot as plt

time = ["6:04","6:16","6:44","7:03","7:19","7:39","8:58","9:09","9:46","16:48","18:36","20:30","20:44","21:00","21:12"]
recall_1_map_0_bow = [0.8936,0.9583,0.7396,0.6989,0.9421,0.95,1,0.9933,0.9871,0.7989,0.8038,0.6989,0.8178,0.7907,0.5634]
recall_1_map_13_bow = [0.6084,0.6993,0.592,0.6344,0.7105,0.6971,0.656,0.702,0.6507,0.6443,0.6038,0.64,0.6629,0.686,1]
recall_1_map_0_maxpool = [0.9277,0.9493,0.6733,0.628,0.9579,0.9725,1,0.9983,0.9871,0.7523,0.7453,0.7453,0.8178,0.8559,0.5924]
recall_1_map_13_maxpool = [0.7149,0.7464,0.6517,0.6839,0.7684,0.7694,0.672,0.7441,0.7224,0.6834,0.5396,0.7432,0.7813,0.8023,1]

def time_to_decimal(time_str):
    hours, minutes = map(int, time_str.split(':'))
    decimal_time = hours + minutes / 60
    return decimal_time

time_values = [time_to_decimal(time_element) for time_element in time]
plt.title("Recall@1 at different times")
plt.plot(time_values,recall_1_map_0_bow,label="map: run 0 (8:58) - BOW")
plt.plot(time_values,recall_1_map_13_bow, label = "map: run 13 (21:12) - BOW")
plt.plot(time_values,recall_1_map_0_maxpool,label="map: run 0 (8:58) - (max-pool)")
plt.plot(time_values,recall_1_map_13_maxpool,label="map: run 13 (21:12) - (max-pool)")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Rate")
plt.savefig("/home/lamlam/code/visual_place_recognition/clustering/results/recall_1_vs_time.png")

