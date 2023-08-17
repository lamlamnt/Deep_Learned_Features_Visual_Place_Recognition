import numpy as np
import sys
sys.path.append("/home/lamlam/downloads/robotcar-dataset-sdk/python")
import transform
from PIL import Image

xyzrpy = np.array([-2.0582, 0.0894, 0.3675, -0.0119, -0.2498, 3.1283])
rear_extrinsic_seasons_sdk = np.asarray(transform.build_se3_transform(xyzrpy))

rear_extrinsic_seasons_theirs = np.linalg.inv(np.array([[-0.999802, -0.011530, -0.016233, 0.060209],
                                   [-0.015184, 0.968893, 0.247013, 0.153691],
                                    [0.012880, 0.247210, -0.968876, -2.086142],
                                    [0.000000, 0.000000, 0.000000, 1.000000]]))
T_C_G = np.array([[0,1,0,0],
                 [0,0,1,0],
                 [1,0,0,0],
                 [0,0,0,1]])
#1425997186158107
run_0_1 = np.array([[0.444545,-0.182161,0.877039, -199.836338], 
                    [0.895183,0.055309,-0.442254,50.954163], 
                    [0.032053, 0.981712, 0.187655, -1.312718], 
                    [0.000000,0.000000,0.000000,1.000000]])
#1425997445935721 
run_0_2 = np.array([[0.927613,0.050644,-0.370094,-307.104803], 
                    [-0.373005,0.178759,-0.910446,391.144300], 
                    [0.020049,0.982589,0.184709,-1.172539], 
                    [0.000000,0.000000,0.000000,1.000000]])
T_G2_G1 = np.linalg.inv(run_0_2)@run_0_1
#print(T_G2_G1)
#print(run_0_1@run_0_2)
#print(transform.so3_to_euler(run_0_1[:3,:3]))
#print(transform.so3_to_euler(run_0_2[:3,:3]))

#Converts Bumblee to INS
ins_extrinsic = np.array([-1.7132,0.1181,1.1948,-0.0125,0.0400,0.0050])
ins_extrinsic_se3 = np.asarray(transform.build_se3_transform(ins_extrinsic))
T_I_V = np.array([[1,0,0,0],
                [0,-1,0,0],
                [0,0,-1,0],
                [0,0,0,1]])
T_camUT_stereRo = np.array([[0,1,0,0],
                            [0,0,1,0],
                            [1,0,0,0],
                            [0,0,0,1]])
script_transformation = T_camUT_stereRo@(np.linalg.inv(ins_extrinsic_se3)@T_I_V)
print(script_transformation)
print(T_camUT_stereRo@(ins_extrinsic_se3@T_I_V))
#print(ins_extrinsic_se3)
#print(np.linalg.inv(ins_extrinsic_se3))

image = Image.open("/Volumes/scratchdata/lamlam/processed_data/robotcar_rtk_full/run_000011/images/left/1417535825194576.png")  # Replace with your image file path
width, height = image.size

print("Width:", width)
print("Height:", height)