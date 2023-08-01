import os

#For each run, get the list of images available and rewrite se3_grasshopper file
root_dir = "/Volumes/scratchdata/lamlam/processed_data/robotcar_seasons"
runs = [0,1,2,3,4,5,6,7,8]
gh_file_name_old = "se3_grasshopper.txt"
gh_file_name_new = "se3_grasshopper_2.txt"

for run in runs:
    path_to_images_folder = os.path.join(root_dir,"run_" + str(run).zfill(6),"images","left")
    images = sorted(os.listdir(path_to_images_folder))
    path_to_gh_old = os.path.join(root_dir,"run_" + str(run).zfill(6),gh_file_name_old)
    path_to_gh_new = os.path.join(root_dir,"run_" + str(run).zfill(6),gh_file_name_new)
    difference = 0
    with open(path_to_gh_old,"r") as input_file, open(path_to_gh_new,"w") as output_file:
        i = 0
        for line in input_file:
            se3_full = line.strip().split()[1:]
            se3_full = [str(element) for element in se3_full]
            content = " ".join(se3_full)
            output_file.write(images[i].rsplit(".", 1)[0] + " " + content + "\n")
            i+=1
