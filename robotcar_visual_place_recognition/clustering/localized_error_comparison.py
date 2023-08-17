import numpy as np
#Every 0.5 meters -> get localized frame
#Assume reference run is run 0
def get_localized_frame(query_run,query_len, config):
    incre = 6
    incre_ref = 2
    if(query_run < 10):
        name = "0" + str(query_run)
    else:
        name = str(query_run)
    transform_path = "/Volumes/oridatastore09/ThirdPartyData/utias/inthedark/run_0000" + name + "/transforms_spatial.txt"
    localized_frames = np.zeros(query_len,dtype=int)
    with open(transform_path, 'r') as file:
        for i in range(query_len):
            line = file.readline().strip()  # Read the first line and remove leading/trailing whitespace
            numbers = line.split(",")  # Split the line by comma
            localized_frames[i] = numbers[3]
            for j in range(incre-1):
                file.readline()
    localized_frames= localized_frames/incre_ref
    localized_frames = np.round(localized_frames).astype(int)
    print(localized_frames)
    return localized_frames

get_localized_frame(16,543,0.5)

