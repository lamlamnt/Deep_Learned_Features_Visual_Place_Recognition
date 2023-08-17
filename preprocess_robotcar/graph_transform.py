import matplotlib.pyplot as plt
import numpy as np

file_path = "/Volumes/scratchdata/lamlam/processed_data/robotcar_seasons/run_000005/transforms_spatial.txt"
#Plot the x and y 
with open(file_path,"r") as input_file:
    x_list = []
    y_list = []
    for line in input_file:
        info = line.strip().split(",")
        x = info[7]
        y = info[11]
        x_list.append(x)
        y_list.append(y)

plt.title("X and y of run 5")
plt.plot(x_list,y_list)
plt.xticks(np.arange(0, 600, step=50))  
plt.yticks(np.arange(0,600,step=50))
plt.savefig("Ground_truth_accuracy.png")
    