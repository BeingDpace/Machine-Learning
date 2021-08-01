import numpy as np
def PCA(X , k):
# k is number of PC-components
    # Subtract the mean of each variable from the dataset so that the dataset should be centered on the origin
    X_mean = X - np.mean(X , axis = 0)
    # calculating the covariance matrix of the mean-centered data.
    covariance_matrix = np.cov(X_mean , rowvar = False)
    # Calculating Eigenvalues and Eigenvectors of the covariance matrix
    eigen_values , eigen_vectors = np.linalg.eigh(covariance_matrix)
    # sort the eigenvalues in descending order
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    # similarly sort the eigenvectors
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
    # select the first k eigenvectors, k is desired dimension of our final reduced data.
    eigenvector_subset = sorted_eigenvectors[:,0:k]
    # Transform the data: transform the data by having a dot product between the
    # Transpose of the Eigenvector subset and the Transpose of the mean-centered data.
    # By transposing the outcome of the dot product, the result we get is the data reduced to lower dimensions from higher dimensions.
    X_final = np.dot(eigenvector_subset.transpose() , X_mean.transpose() ).transpose()
    return X_final
