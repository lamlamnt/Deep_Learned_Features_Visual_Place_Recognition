import numpy as np
import sys
sys.path.append("/home/lamlam/downloads/robotcar-dataset-sdk/python")
import transform

#From gps to se3
def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])

def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])

def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

def pose_from_oxts_packet(lat, lon, alt, roll, pitch, yaw):

    """Helper method to compute a SE(3) pose matrix from an OXTS packet.
    Taken from https://github.com/utiasSTARS/pykitti
    """
    scale = np.cos(lat * np.pi / 180.)

    er = 6378137.  # earth radius (approx.) in meters
    # Use a Mercator projection to get the translation vector
    #ty = lat * np.pi * er / 180.

    tx = scale * lon * np.pi * er / 180.
    ty = scale * er * np.log(np.tan((90. + lat) * np.pi / 360.))
    tz = alt
    t = np.array([tx, ty, tz]).reshape(-1,1)

    # Use the Euler angles to get the rotation matrix
    Rx = rotx(roll)
    Ry = roty(pitch)
    Rz = rotz(yaw)
    R = Rz.dot(Ry.dot(Rx))
    return transform_from_rot_trans(R, t)

#Seasons se3 data - translation unit is in m
#1416907251866820 
se3_gh_1 = np.array([[0.310355,-0.161433,0.936813,-95.890304],
        [0.948644, 0.116117, -0.294264, 10.778036], 
        [-0.061276, 0.980029, 0.189180, -0.904008], 
        [0.000000, 0.000000, 0.000000, 1.000000]])

#1416907617176924 
se3_gh_2 = np.array([[0.040894, -0.199100, 0.979126, -491.663926], 
                    [0.999030, -0.007881, -0.043328, 680.101094], 
                    [0.016343, 0.979947, 0.198585, -1.085389], 
                    [0.000000, 0.000000, 0.000000, 1.000000]])

rear_extrinsic_seasons = np.array([[-0.999802, -0.011530, -0.016233, 0.060209],
                                   [-0.015184, 0.968893, 0.247013, 0.153691],
                                    [0.012880, 0.247210, -0.968876, -2.086142],
                                    [0.000000, 0.000000, 0.000000, 1.000000]])

ins = np.array([-1.7132,0.1181, 1.1948, -0.0125, 0.0400, 0.0050])
se3_ins = transform.build_se3_transform(ins)
inv_se3_ins_extrinsic = np.linalg.inv(se3_ins)

def convert_gh_to_stereo(se3_gh, extrinsic):
    return se3_gh@extrinsic

def convert_gh_to_stereo_inv(se3_gh,extrinsic):
    return se3_gh@np.linalg.inv(extrinsic)

#frame 2 is the reference frame
def get_relative(frame1,frame2):
    return np.dot(frame2, np.linalg.inv(frame1))

#RTK
#1416907251899780
lat1, lon1, alt1, roll1, pitch1, yaw1 = 51.7597380555,-1.2612197491,108.813591,0.0050046706674882965,-0.012750565345449098,-1.8537263543425386

#1416907617187710
lat2, lon2, alt2, roll2, pitch2, yaw2 = 51.7560381124,-1.251678262,108.382408,0.03598725661510903,0.004718245528682865,-1.5911974757055787

#Sample - translation unit also in m
se3_sample_utias = np.array([[0.99995,-0.00674617,-0.007411,0.16172],
                             [0.006793,0.999957,0.006239,0.047683],
                             [0.007369,-0.00629,0.99995,-0.01652],
                             [0.0,0.0,0.0,1.0]])

#Using the other extrinsic matrix (from robotcar sdk)
xyzrpy = np.array([-2.0582, 0.0894, 0.3675, -0.0119, -0.2498, 3.1283])
se3_calculated = transform.build_se3_transform(xyzrpy)
inv_se3_calculated_extrinsic = np.linalg.inv(se3_calculated)

np.set_printoptions(suppress=True, precision=4)

#Correct rotation matrix
#print(transform.euler_to_so3([roll1,pitch1,yaw1]))

gps_se3_1 = pose_from_oxts_packet(lat1,lon1,alt1,roll1,pitch1,yaw1)
gps_se3_2 = pose_from_oxts_packet(lat2,lon2,alt2,roll2,pitch2,yaw2)

seasons_se3_1 = convert_gh_to_stereo(se3_gh_1,rear_extrinsic_seasons)
seasons_se3_2 = convert_gh_to_stereo(se3_gh_2,rear_extrinsic_seasons)

seasons_se3_1_inv = convert_gh_to_stereo_inv(se3_gh_1,rear_extrinsic_seasons)
seasons_se3_2_inv = convert_gh_to_stereo_inv(se3_gh_2,rear_extrinsic_seasons)

seasons_se3_sdk_1 = convert_gh_to_stereo(se3_gh_1,inv_se3_calculated_extrinsic)
seasons_se3_sdk_2 = convert_gh_to_stereo(se3_gh_2,inv_se3_calculated_extrinsic)

gps_se3_1_converted = convert_gh_to_stereo(gps_se3_1,inv_se3_ins_extrinsic)
gps_se3_2_converted = convert_gh_to_stereo(gps_se3_2,inv_se3_ins_extrinsic)

#print(seasons_se3_1)
#For some reasons this gives much larger translation distance
print(get_relative(gps_se3_1_converted,gps_se3_2_converted))
#print(seasons_se3_1_inv)
#print(seasons_se3_1)
#print(get_relative(seasons_se3_1_inv,seasons_se3_2_inv))
#print(get_relative(seasons_se3_1,seasons_se3_2))
#print(get_relative(seasons_se3_sdk_1,seasons_se3_sdk_2))
#The last 2 give the exact same matrix. 

#Convert COLMAP quaternions to se3_gh 
#1417176579528074
colmap1 = [-0.115306, -0.0569793, 0.648813, 0.75, -39.2066, -10.2264, 10.6255] 
#1417176579577144
lat3, lon3, alt3, roll3, pitch3, yaw3 = 51.7602239312,-1.2614872236,111.832779,0.03159227432950274,-0.0057518025987368425,-1.9265392719510237

#1417178873750232
colmap2 = [0.444353, -0.252, 0.617843, 0.597759, -13.0622, 0.2453, 36.8999] 
#1417178873398565
lat4, lon4, alt4, roll4, pitch4, yaw4 = 51.7608284638,-1.2617708676,111.792206,0.031282559745796835,0.0013314107033412827,2.956120307957885
gps_se3_4 = pose_from_oxts_packet(lat4,lon4,alt4,roll4,pitch4,yaw4)



