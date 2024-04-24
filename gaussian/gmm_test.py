

from sklearn.mixture import GaussianMixture
import numpy as np

# Assuming you have a 1D numpy array of float values named "data"
# For the sake of demonstration, let's simulate some data that would resemble your description
np.random.seed(42)  # For reproducibility

# Simulating data from 5 different Gaussian distributions
data1 = np.random.normal(loc=-5, scale=2, size=20000)
data2 = np.random.normal(loc=0, scale=1, size=20000)
data3 = np.random.normal(loc=5, scale=3, size=20000)
data4 = np.random.normal(loc=10, scale=2, size=20000)
data5 = np.random.normal(loc=15, scale=1.5, size=20000)

# Concatenating the data to form a single dataset
data = np.concatenate([data1, data2, data3, data4, data5])

# Reshaping data for sklearn's GMM
data = data.reshape(-1, 1)

np.random.shuffle(data)

# Creating and fitting the GMM
gmm = GaussianMixture(n_components=5, random_state=42)
gmm.fit(data)

# The means, variances, and weights of the Gaussians found by GMM
means = gmm.means_
covariances = gmm.covariances_
weights = gmm.weights_

# Printing the parameters for each Gaussian component
print("Means:", means.flatten())
print("Variances:", covariances.flatten())
print("Weights:", weights)

