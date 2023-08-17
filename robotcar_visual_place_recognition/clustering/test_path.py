import glob

def path_process_ref_que(run):
    name = ""
    if(run < 10):
        name = "0" + str(run)
    else:
        name = str(run)
    images_path = glob.glob("/Volumes/oridatastore09/ThirdPartyData/utias/inthedark/run_0000" + name + "/images/left/*.png")
    gps_path ="/Volumes/oridatastore09/ThirdPartyData/utias/inthedark/run_0000" + name + "/gps.txt"
    with open(gps_path, 'r') as file:
        line_count = 0
        for line in file:
            line_count += 1
    #Some images at the end are missing gps data
    difference = len(images_path) - line_count
    if(difference > 0):
        images_path = images_path[:-difference]
    #Downsampled
    images_path = sorted(images_path)
    new_list = [str(i).zfill(6) for i in range(0,len(images_path),6)]
    print('\n'.join(new_list))
    return images_path, len(images_path)

path_process_ref_que(16)