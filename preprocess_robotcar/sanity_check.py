import numpy as np
import sys
sys.path.append("/home/lamlam/downloads/robotcar-dataset-sdk/python")
import transform

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

xyzrpy = np.array([-2.0582, 0.0894, 0.3675, -0.0119, -0.2498, 3.1283])
rear_extrinsic_different = np.linalg.inv(transform.build_se3_transform(xyzrpy))
rear_extrinsic_not_inv = transform.build_se3_transform(xyzrpy)

new_ans = np.linalg.inv(rear_extrinsic)@(np.linalg.inv(gh_0)@(gh_5@rear_extrinsic))

different_ans = np.linalg.inv(rear_extrinsic_different)@(np.linalg.inv(gh_0)@(gh_5@rear_extrinsic_different))

#Run 0 - 5
current_ans = np.array([[-0.06421468323456465,0.11301567238002568,-0.9915155586475402,2.7511739797226937],
                       [-0.08485110936986243,0.9893499964747207,0.11826493382810375,6.091877130751797],
                       [0.9943220892732083,0.0917265305245002,-0.053942475412332956,-98.1778693826783],
                       [0.0,0.0,0.0,1.0]])


#5 - 1432293482831071
#0 - 1425997174159731
current_ans_2 = np.array([[0.9997398252608762,0.0015171667706923887,-0.022729894488354472,-0.07980631963760842],
                          [-0.0016423893141883099,0.999982980921042,-0.005463925494202774,0.23067597674665308],
                          [0.02272155862719848,0.005500281693894245,0.9997260579772737,-2.5871252448632305],
                          [0.0,0.0,0.0,1.0]])
gh_0_2 =np.array([[0.304358, -0.157201, 0.939497, -101.929334],
                [0.951190, 0.102999, -0.290912, 12.579252], 
                [-0.051036, 0.982181, 0.180877, -1.058650], 
                [0.000000,0.000000,0.000000,1.000000]])
gh_5_2 = np.array([[0.326042, -0.160794, 0.931580, -99.437655],
                  [0.943987, 0.108371, -0.311679, 11.860682],
                  [-0.050840,0.981020,0.187121,-0.994980], 
                  [0.000000,0.000000,0.000000,1.000000]])
np.set_printoptions(suppress=True)
#Old answer
#print(rear_extrinsic@(np.linalg.inv(gh_0_2)@(gh_5_2@np.linalg.inv(rear_extrinsic))))
#print(np.linalg.inv(np.linalg.inv(rear_extrinsic)@(np.linalg.inv(gh_0_2)@(gh_5_2@rear_extrinsic))))

#New answer - but using the inverse
#print(np.linalg.inv(rear_extrinsic_different)@(np.linalg.inv(gh_0_2)@(gh_5_2@rear_extrinsic_different)))

#New answer - but not using the inverse
#print(np.linalg.inv(rear_extrinsic_not_inv)@(np.linalg.inv(gh_0_2)@(gh_5_2@rear_extrinsic_not_inv)))

#This is P_G1 and P_G2
dummy_1 = np.array([[1,0,0,1],
                    [0,1,0,1],
                    [0,0,1,1],
                    [0,0,0,1]])
dummy_2 = np.array([[1,0,0,10],
                    [0,1,0,5],
                    [0,0,1,7],
                    [0,0,0,1]])
xyzrpy = np.array([-2.0582, 0.0894, 0.3675, -0.0119, -0.2498, 3.1283])
rear_extrinsic_final = transform.build_se3_transform(xyzrpy)
print(rear_extrinsic_final)
#This is B2_B1
T_b = np.linalg.inv(rear_extrinsic_final)@(np.linalg.inv(dummy_2)@(dummy_1@rear_extrinsic_final))
print(T_b)

#Point in B1 frame
dummy_b1 = np.array([[1],[1],[1],[1]])
print(T_b@dummy_b1)

rear_extrinsic_final = np.linalg.inv(rear_extrinsic)
T_b_2 = np.linalg.inv(rear_extrinsic_final)@(np.linalg.inv(dummy_2)@(dummy_1@rear_extrinsic_final))
print(T_b_2@dummy_b1)

T_b_old = np.linalg.inv(rear_extrinsic)@(np.linalg.inv(dummy_2)@(dummy_1@rear_extrinsic))
print(T_b_old@dummy_b1)


#Compare matrices
#0,1425997174472195,0,1425997185283244 
old = np.array([[0.9872899477238437,-0.07273194631025359,0.14131287718515295,-6.587058116441888],
                [0.07278189054609176,0.9973365408908044,0.0048223253221036325,5.720085985712474],
                [-0.14128655454994518,0.005524083498557071,0.9899536252325669,-93.94441745255911],
                [0.0,0.0,0.0,1.0]])
new = np.array([[0.9872967238066024,0.0378735013049417,-0.154308662570292,14.491812893122496],
                [-0.03729361606403029,0.9992825486826376,0.006652412990329454,16.277580503010178],
                [0.1544491375301982,-0.0008130636838704317,0.9880008413579754,92.11911496314501],
                [0.0,0.0,0.0,1.0]])
print(old@dummy_b1)
print(new@dummy_b1)

T_s_v = np.array([[0.5,0.3,0,0],
                          [0,0.9,0,0],
                          [0,0.2,0.6,1.52],
                          [0,0,0,1]])
print(np.linalg.inv(T_s_v)@(dummy_1@T_s_v))



