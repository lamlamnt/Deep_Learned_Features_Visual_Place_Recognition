import os
import shutil

pose_path = "/Volumes/scratchdata/lamlam/robotcar_v2_train.txt"
pose_path_v1 = "/Volumes/scratchdata/lamlam/robotcar_v2_train_v1.txt"
file_name = "se3_grasshopper.txt"
runs = {"dawn":"2014-12-16-09-14-09","dusk":"2015-02-20-16-34-06", "night":"2014-12-10-18-10-50", 
        "night-rain":"2014-12-17-18-18-43", "overcast-summer":"2015-05-22-11-14-30", "overcast-winter":"2015-11-13-10-28-08", 
        "rain":"2014-11-25-09-18-32", "snow":"2015-02-03-08-45-10", "sun":"2015-03-10-14-18-10"}
runs_time = ["2014-12-16-09-14-09","2015-02-20-16-34-06","2014-12-10-18-10-50","2014-12-17-18-18-43","2015-05-22-11-14-30",
            "2015-11-13-10-28-08","2014-11-25-09-18-32","2015-02-03-08-45-10","2015-03-10-14-18-10"]

def sort_create():
#Sort the lines in the train.txt file and create a se3_grasshopper.txt for each run (in the run's folder)
    with open(pose_path, "r") as file:
        lines = sorted(file.readlines())
    with open(pose_path_v1, "w") as file:
        file.writelines(lines)
    entries = os.listdir("/Volumes/scratchdata/lamlam/processed_data/robotcar_seasons")
    for entry in entries:
        entry_path = os.path.join("/Volumes/scratchdata/lamlam/processed_data/robotcar_seasons", entry)
        file_path = os.path.join(entry_path, file_name)
        # If the file already exists, it will be truncated (content will be deleted)
        with open(file_path, "w") as new_file:
            new_file.write("")

def write_to_file():
#Copy the data to each txt file in each folder
    f = open(pose_path,"r")
    with open(pose_path_v1, "r") as file:
        current_run = ""
        for line in file:
            parts = line.strip().split("/")
            name_of_run = parts[0]
            #If new run then open new file
            if(name_of_run != current_run):
                f.close()
                current_run = name_of_run
                se3_path = os.path.join("/Volumes/scratchdata/lamlam/processed_data",name_of_run + "_" + runs[name_of_run],file_name)
                f = open(se3_path,"w")
            f.write(parts[2] + "\n")
        
#Check to make sure that the number of lines match up 
#robotcar_v2_train_v1.txt has 1906 lines
def get_line_count():
#Print the number of lines in each file in the directory
    entries = os.listdir("/Volumes/scratchdata/lamlam/processed_data")
    for entry in entries:
        entry_path = os.path.join("/Volumes/scratchdata/lamlam/processed_data", entry)
        file_path = os.path.join(entry_path, file_name)
        line_count = sum(1 for line in open(file_path,"r"))
        print(file_path)
        print(line_count)

def remove_jpg():
#Remove the .jpg from the first entry of each line
    entries = os.listdir("/Volumes/scratchdata/lamlam/processed_data")
    for entry in entries:
        entry_path = os.path.join("/Volumes/scratchdata/lamlam/processed_data", entry)
        file_path = os.path.join(entry_path, file_name)
        with open(file_path,"r") as f:
            newline=[]
            for word in f.readlines():        
                newline.append(word.replace(".jpg",""))
        with open(file_path,"w") as f:
            for line in newline:
                f.writelines(line)

#Just change the runs in runs_time if just want to do it for a specific leftover run (to avoid shutil.copy giving an error)
def get_closest():
#Returns a list, which contains 9 sublists, with each sublist containing the frame number closest to the ones in se3 file
#Copies theses frames into the new folder
    for run in runs_time:
        #Get the path of each folder (stereo_left)
        folder_path = os.path.join("/Volumes/scratchdata/lamlam",run,"stereo/left")
        entries = sorted(os.listdir(folder_path))
        #entries is a list of the names of all the images under stereo_left for a run
        #strip, just get the number, and cast to an int
        images_timestamps = [name.rstrip(".png") for name in entries]
        images_timestamps = sorted([int(element) for element in images_timestamps])
        
        #Get target number in se3 file
        file_path = os.path.join("/Volumes/scratchdata/lamlam/processed_data/robotcar_seasons",next((key for key, value in runs.items() if value == run), None)
 + "_" + run,file_name)
        chosen_frames = []
        with open(file_path,"r") as f:
            for line in f:
                target_num = line.strip().split()[0]
                chosen_num = find_closest_number(images_timestamps,int(target_num))
                chosen_frames.append(chosen_num)
        #Create folder left, right 
        existing_folder_path = os.path.join("/Volumes/scratchdata/lamlam/processed_data/robotcar_seasons",next((key for key, value in runs.items() if value == run), None)
 + "_" + run)
        new_folder_path_left = os.path.join(existing_folder_path, "images/left")
        new_folder_path_right = os.path.join(existing_folder_path, "images/right")
        new_folder_path_center = os.path.join(existing_folder_path, "images/centre")
        #If the folder already exists, makedirs won't do anything
        os.makedirs(new_folder_path_left)
        os.makedirs(new_folder_path_right)
        os.makedirs(new_folder_path_center)
        for frame in chosen_frames:
            #Copy the chosen frames to the processed data folder. If they already exist, will give an error
            source_file_left = os.path.join("/Volumes/scratchdata/lamlam",run,"stereo/left",str(frame) + ".png")
            source_file_right = os.path.join("/Volumes/scratchdata/lamlam",run,"stereo/right",str(frame) + ".png")
            source_file_center = os.path.join("/Volumes/scratchdata/lamlam",run,"stereo/centre",str(frame) + ".png")
            shutil.copy(source_file_left, new_folder_path_left)
            shutil.copy(source_file_right, new_folder_path_right)
            shutil.copy(source_file_center, new_folder_path_center)

#Uses binary search
def find_closest_number(sorted_numbers, target):
    left, right = 0, len(sorted_numbers) - 1
    closest_number = None
    while left <= right:
        mid = (left + right) // 2
        current_number = sorted_numbers[mid]
        if closest_number is None or abs(current_number - target) < abs(closest_number - target):
            closest_number = current_number
        if current_number == target:
            return current_number
        elif current_number < target:
            left = mid + 1
        else:
            right = mid - 1
    return closest_number

def parse_gps():
#If rtk exists, use rtk. Otherwise, use gps/ins
    pass


if __name__ == '__main__':






