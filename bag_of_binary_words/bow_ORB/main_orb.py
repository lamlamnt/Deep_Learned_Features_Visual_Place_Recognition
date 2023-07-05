import glob
import cv2
import numpy as np
import sys
sys.path.append("/home/lamlam/code/bag_of_binary_words/PyDBoW")
import dbow
from haversine import haversine
np.random.seed(123)

def createDatabase(images_path, vocabulary_path, database_path):
    print("Loading images and creating vocabulary")
    images = []
    for image_path in images_path:
        images.append(cv2.imread(image_path))
    vocabulary = dbow.Vocabulary(images, n_clusters, depth)
    vocabulary.save(vocabulary_path)
    orb = cv2.ORB_create()

    print("Creating Bag of Binary Words from Images")
    bows = []
    for image in images:
        kps, descs = orb.detectAndCompute(image, None)
        descs = [dbow.ORB.from_cv_descriptor(desc) for desc in descs]
        bows.append(vocabulary.descs_to_bow(descs))

    print("Creating Database")
    db = dbow.Database(vocabulary)
    for image in images:
        kps, descs = orb.detectAndCompute(image, None)
        descs = [dbow.ORB.from_cv_descriptor(desc) for desc in descs]
        db.add(descs)
    db.save(database_path)
    return db

def path_processing(reference_run, every_n_image):
    ref_name = ""
    if(reference_run < 10):
        ref_name = "0" + str(reference_run)
    else:
        ref_name = str(reference_run)
    images_path = glob.glob("/Volumes/oridatastore09/ThirdPartyData/utias/inthedark/run_0000" + ref_name + "/images/left/*.png")
    downsampled = sorted(images_path)[::every_n_image]

    vocabulary_path = "/home/lamlam/data/bow/vocabulary_inthedark_run" + ref_name + ".pickle"
    database_path = "/home/lamlam/data/bow/database_inthedark_run" + ref_name + ".pickle"
    return downsampled,vocabulary_path, database_path

def query_image(query_run, db):
    if(query_run < 10):
        que_name = "0" + str(query_run)
    else:
        que_name = str(query_run)
    query_path = glob.glob("/Volumes/oridatastore09/ThirdPartyData/utias/inthedark/run_0000" + que_name + "/images/left/*.png")
    downsampled = sorted(query_path)[::200]
    images = []
    for image_path in downsampled:
        images.append(cv2.imread(image_path))
    index = []
    for image in images:
        orb = cv2.ORB_create()
        kps, descs = orb.detectAndCompute(image, None)
        descs = [dbow.ORB.from_cv_descriptor(desc) for desc in descs]
        scores = db.query(descs)
        #match_bow = db[np.argmax(scores)] #The bow that matches the best
        #match_desc = db.descriptors[np.argmax(scores)]
        index.append(np.argmax(scores))
        #kps and descs are 487. Scores = same as input size. Len of db same as len of inputs into create database function
        #The descriptors and bow are stored in order 
    return index

#Get ground truth from GPS data (need to set a threshold) OR if based on localized spatial data (from run 0)
#Using GPS allows us to quantify the error instead of just having 0 or 1 success
#Only works for reference run 0 for now
def get_ground_truth(query_run,index):
    #gps in the form latitude, longitude, rotation
    #Uses the haversine formula to find the distance between 2 gps points
    #For each image in the 
    if(query_run < 10):
        que_name = "0" + str(query_run)
    else:
        que_name = str(query_run)
    gps_path = "/Volumes/oridatastore09/ThirdPartyData/utias/inthedark/run_0000" + que_name + "gps.txt"
    with open(gps_path, "r") as file:
        for i in range(30):
            line = file.readline().strip()  # Read the first line and remove leading/trailing whitespace
            numbers = line.split(",")  # Split the line by comma
            lat2 = numbers[2]
            lon2 = numbers[3]
    #distance = haversine((lat1, lon1), (lat2, lon2))

#Set parameters:
similarity_threshold = 0.8
depth = 2 #6
n_clusters = 10
every_n_image = 200 #Downsample framerate from around 1115 images (260 meter path)
reference_run = 0
query_run = 1
#Different runs of the same path
gps_paths = [0,1,2,5,6,9,10,11,12,13,14,15,16,17,18,19,25,26,27,28,29,30,31]

#Create database from 1 run (and save to files)
images_path, vocabulary_outpath,database_outpath = path_processing(reference_run, every_n_image)
db = createDatabase(images_path, vocabulary_outpath, database_outpath)
print("Finished creating database")

#Load database from files
#loadDatabase(vocabulary_path, database_path)

#Query each image from a different run 
query_image(query_run, db)

#Get the candidate image with the highest score and calculate gps difference 
#If above the similarity_threshold then yes:loop closure 

#Calculate accuracy rate using ground truth

