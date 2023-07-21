import numpy as np

def gh_to_stereo(se3):
    #Might want inverse of this
    rear_extrinsic_seasons = np.array([[-0.999802, -0.011530, -0.016233, 0.060209],
                                   [-0.015184, 0.968893, 0.247013, 0.153691],
                                    [0.012880, 0.247210, -0.968876, -2.086142],
                                    [0.000000, 0.000000, 0.000000, 1.000000]])
    return se3@rear_extrinsic_seasons

#frame 2 is the reference frame
#Maybe switch position of in the matrix multiplication
def get_relative(frame1,frame2):
    return np.dot(frame2, np.linalg.inv(frame1))

def extract_translation(se3_matrix):
    # Extract the translational part (last column) from the SE(3) matrix
    translation = se3_matrix[:3, 3]
    return translation

def euclidean_distance(point1, point2):
    # Calculate the Euclidean distance as the square root of the squared differences
    squared_diff = np.sum((point1 - point2) ** 2)
    distance = np.sqrt(squared_diff)
    return distance

def find_closest_se3_index(se3_query,list_ref):
    #Returns the index of the closest se3 in list_ref

    # Extract the translational part from the query matrix
    translation_1 = extract_translation(se3_query)

    # Calculate the Euclidean distance between the translational parts of se3_matrix_1 and the list of matrices
    distances = [euclidean_distance(translation_1, extract_translation(se3_matrix)) for se3_matrix in list_ref]

    min_value = min(distances)
    min_index = distances.index(min_value)
    return min_index


