import numpy as np
import matplotlib.pyplot as plt

#Size {544,543}
A_B = np.loadtxt("/home/lamlam/code/visual_place_recognition/clustering/similarity_matrix.txt",dtype=float)
max_magnitude_indices = np.argmin(A_B, axis=0)
num_removed = 400 #Have to get around 400 to actually see a difference to the original matrix 
U, S, VT = np.linalg.svd(A_B, full_matrices=False)
#Already sorted in descending order 
compressed_data = U[:, :-num_removed] @ np.diag(S[:-num_removed]) @ VT[:-num_removed, :]
print(max_magnitude_indices)
max_compressed_indices = np.argmin(compressed_data,axis=0)
print(max_compressed_indices)

#Plot the similarity matrix
plt.figure()
plt.title("Similarity between runs")
similarity_plot = plt.imshow(compressed_data, cmap='viridis', interpolation='nearest')
colorbar = plt.colorbar(similarity_plot)
colorbar.set_label("cross_entropy")
plt.xlabel("Query run")
plt.ylabel("Reference run")
plot_name = "Compressed_data.png"
plt.savefig("/home/lamlam/code/visual_place_recognition/clustering/plot_visualize_cluster/" + plot_name)

plt.figure()
plt.title("Reference frame localised to - compressed")
plt.plot(max_compressed_indices)
plt.xlabel("Query run")
plt.ylabel("Reference run")
plot_name = "Compressed_localisation.png"
plt.savefig("/home/lamlam/code/visual_place_recognition/clustering/plot_visualize_cluster/" + plot_name)

"""
#S is a row vector. U has the larger size (544,544). S has size (543). VT has size (543,543)
#Full matrix multiplication means S would be padded to have size (544,543) -> extra row at the bottom
sorted_indices = np.argsort(np.abs(S))[::-1][:-num_removed]
reduced_singular = S[sorted_indices]  #In descending order 
reduced_U = U[sorted_indices, sorted_indices]
reduced_VT = VT[sorted_indices,sorted_indices]
#Padding of singular value matrix
#S goes down to size (542) -> Reduced_U is {543,543}. S is (543,542). VT is (542,542)
#S needs to be the shape {542,543}

reconstructed_matrix = np.dot(reduced_U,np.dot(np.diag(reduced_singular),reduced_VT))
print(reconstructed_matrix)
reconstructed_matrix_2 = sum(reduced_singular[i] * np.outer(reduced_U[:, i], reduced_VT[:, i]) for i in range(len(reduced_singular)))
print(reconstructed_matrix.shape)
print(A_B)
print(reconstructed_matrix)
#Contains complex values
max_magnitude_reconstructed = np.argmin(np.abs(reconstructed_matrix), axis=1)
"""