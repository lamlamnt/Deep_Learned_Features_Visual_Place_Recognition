import numpy as np
import matplotlib.pyplot as plt

#Open file and load data
matrix = np.loadtxt("/home/lamlam/code/visual_place_recognition/clustering/similarity_matrix.txt",dtype=float)
#Make the matrix square 
matrix = matrix[:-1,:]
eigenvalues, eigenvectors = np.linalg.eig(matrix)
#Eigenvalues are positive/negative and complex. Should sort based on magnitude 

sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]  #Eigenvalues in descending order 
#total_entropy = -np.sum(sorted_eigenvalues * np.log2(sorted_eigenvalues))
y_value = []
max_entropy_index = 0

for k in range(1, len(sorted_eigenvalues)):
    #Take the larger eigenvalues
    reduced_eigenvalues = sorted_eigenvalues[:-k]
    entropy = -np.sum(reduced_eigenvalues * np.log2(reduced_eigenvalues))
    y_value.append(entropy)

print(y_value.index(max(y_value)))
plt.figure()
plt.title("Entropy of similarity matrix")
plt.plot(y_value)
plt.xlabel("Number of Eigenvalues Removed")
plt.ylabel("Entropy")
plt.savefig("/home/lamlam/code/visual_place_recognition/clustering/plot_visualize_cluster/entropy.png")

"""
#Plot the eigenvalues 
max_magnitude_indices = np.argmin(matrix, axis=1)
num_removed = 1
eigenvalues, eigenvectors = np.linalg.eig(matrix)
sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]  #In descending order 

reduced_eigenvalues = sorted_eigenvalues[:-num_removed]
sorted_eigenvectors = eigenvectors[:, sorted_indices]
reconstructed_matrix = np.dot(sorted_eigenvectors[:, :num_removed], np.diag(reduced_eigenvalues[:num_removed])).dot(sorted_eigenvectors[:, :num_removed].T)
max_magnitude_indices = np.argmin(np.abs(reconstructed_matrix), axis=1)

#Eigenvectors don't have to be orthogonal
#reconstructed_matrix = np.dot(eigenvectors,np.diag(eigenvalues).dot(np.linalg.inv(eigenvectors)))
#print(reconstructed_matrix)

#reduced_eigenvalues = sorted_eigenvalues[:-num_removed]
#sorted_indices_removed = sorted_indices[:-num_removed]
reduced_eigenvalues = sorted_eigenvalues[:]
sorted_indices_removed = sorted_indices[:]
reduced_eigenvectors = eigenvectors[:, sorted_indices_removed]
reconstructed_matrix = sum(reduced_eigenvalues[i] * np.outer(reduced_eigenvectors[:, i], reduced_eigenvectors[:, i]) for i in range(len(reduced_eigenvalues)))
#reconstructed_matrix = np.dot(sorted_eigenvectors_removed, np.diag(reduced_eigenvalues)).dot(np.linalg.inv(sorted_eigenvectors_removed))
#max_magnitude_indices = np.argmin(np.abs(reconstructed_matrix), axis=1)

#print(matrix)
#print(reconstructed_matrix)
#print(max_magnitude_indices)

matrix_test = np.array([[1,2,3],[4,5,6],[7,8,9]])
eigenvalues, eigenvectors = np.linalg.eig(matrix_test)
sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
reduced_eigenvalues = sorted_eigenvalues[:]
sorted_indices_removed = sorted_indices[:]
reduced_eigenvectors = eigenvectors[:, sorted_indices_removed]
array = np.zeros((3,3))
for i in range(len(reduced_eigenvalues)):
    print(reduced_eigenvectors[:,i])
    print(np.outer(reduced_eigenvectors[:,i],reduced_eigenvectors[:,i]))
    print(reduced_eigenvalues[i])
    print(reduced_eigenvalues[i]* np.outer(reduced_eigenvectors[:,i],reduced_eigenvectors[:,i]))
    array += reduced_eigenvalues[i]* np.outer(reduced_eigenvectors[:,i],reduced_eigenvectors[:,i])
    print(array)

matrix_test2 = np.array([[1,2,3],[4,5,6],[7,8,9]])
u, s, vt = np.linalg.svd(matrix_test2)
"""
