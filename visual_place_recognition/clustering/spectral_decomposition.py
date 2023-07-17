import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from scipy.stats import entropy

#Cross-entropy is not symmetric. -> Must use wasserstein distance 
#Wasserstein distance does not care about spatial relationship
a = [1,2,3]
b = [5,8,9]
print(entropy(a,b))
print(entropy(b,a))
print(wasserstein_distance(a,b))
print(wasserstein_distance(b,a))

#Low means similar
#A is reference run. B is query run
A = np.loadtxt("/home/lamlam/code/visual_place_recognition/clustering/A.txt",dtype=float)
B = np.loadtxt("/home/lamlam/code/visual_place_recognition/clustering/B.txt",dtype=float)
A_B = np.loadtxt("/home/lamlam/code/visual_place_recognition/clustering/similarity_matrix.txt",dtype=float)

#Concatenate matrices
left = np.vstack((A_B.T,A))
right = np.vstack((B,A_B))
full = np.hstack((left,right))
print(np.all(A == A.T))
print(np.all(B == B.T))
print(np.all(full == full.T))
#full has size {1087,1087}

max_magnitude_indices = np.argmin(full, axis=1)
num_removed = 1
eigenvalues, eigenvectors = np.linalg.eig(full)
sorted_indices = np.argsort(np.abs(eigenvalues))[::-1][:-num_removed]
reduced_eigenvalues = eigenvalues[sorted_indices]  #In descending order 
reduced_eigenvectors = eigenvectors[:, sorted_indices]
reconstructed_matrix = sum(reduced_eigenvalues[i] * np.outer(reduced_eigenvectors[:, i], reduced_eigenvectors[:, i]) for i in range(len(reduced_eigenvalues)))
#Contains complex values
max_magnitude_reconstructed = np.argmin(np.abs(reconstructed_matrix), axis=1)

#Plot for sanity check
plt.figure()
plt.title("AB vs AB")
similarity_plot = plt.imshow(full, cmap='viridis', interpolation='nearest')
colorbar = plt.colorbar(similarity_plot)
colorbar.set_label("EMD")
plt.xlabel("AB")
plt.ylabel("AB")
plot_name = "AB.png"
plt.savefig("/home/lamlam/code/visual_place_recognition/clustering/plot_visualize_cluster/" + plot_name)

plt.figure()
plt.title("AB vs AB")
similarity_plot = plt.imshow(reconstructed_matrix, cmap='viridis', interpolation='nearest')
colorbar = plt.colorbar(similarity_plot)
colorbar.set_label("cross_entropy")
plt.xlabel("AB")
plt.ylabel("AB")
plot_name = "Reconstructed.png"
plt.savefig("/home/lamlam/code/visual_place_recognition/clustering/plot_visualize_cluster/" + plot_name)