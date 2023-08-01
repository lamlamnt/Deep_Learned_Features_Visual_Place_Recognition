import numpy as np

#Run 0 - 1425997551358969 
gh_0 = np.array([[-0.998834, -0.047705, 0.007406, -600.611152],
                  [0.016156, -0.185757, 0.982463, 585.633184], 
                  [-0.045493, 0.981437, 0.186312, -1.019885],
                   [0.000000, 0.000000, 0.000000, 1.000000]])
#Run 5 -  1432293804975465 
gh_5 = np.array([[0.079330, -0.167049, 0.982752, -595.804084], 
                [0.996780, 0.001784, -0.080160, 681.693294], 
                [0.011637, 0.985947, 0.166653, -1.048838],
                [0.000000, 0.000000, 0.000000, 1.000000]])

rear_extrinsic = np.array([[-0.999802, -0.011530, -0.016233, 0.060209],
                            [-0.015184, 0.968893, 0.247013, 0.153691],
                            [0.012880, 0.247210, -0.968876, -2.086142],
                            [0.000000, 0.000000, 0.000000, 1.000000]])

new_ans = np.linalg.inv(rear_extrinsic)@(np.linalg.inv(gh_0)@(gh_5@rear_extrinsic))
print(new_ans)

#Run 0 - 5
current_ans = np.array([[-0.06421468323456465,0.11301567238002568,-0.9915155586475402,2.7511739797226937],
                       [-0.08485110936986243,0.9893499964747207,0.11826493382810375,6.091877130751797],
                       [0.9943220892732083,0.0917265305245002,-0.053942475412332956,-98.1778693826783],
                       [0.0,0.0,0.0,1.0]])

print(current_ans)