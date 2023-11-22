import time
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale

def distance(x, X):
    """
    Compute the distance between a given point and all other points
    that are within a specific radius. Consider radius of +inf.
    """
    # X is an array of shape (N, 3), where N is the number of points
    # x is a point of shape (3,)
    # return an array of shape (N,) containing the distance between x and each point in X
    return np.linalg.norm(X - x, axis=1)

def gaussian(dist, bandwidth):
    """
    Compute the weights of points according to the distance computed
    by the distance function.
    """
    # dist is an array of shape (N,) containing the distances between a point and all other points
    # bandwidth is a scalar representing the bandwidth parameter
    # return an array of shape (N,) containing the weights computed using the gaussian function
    return np.exp(-0.5 * (dist / bandwidth) ** 2)

def update_point(weight, X):
    """
    Update the point position according to weights computed
    from the gaussian function.
    """
    # weight is an array of shape (N,) containing the weights computed using the gaussian function
    # X is an array of shape (N, 3) representing the points
    # return the updated point position
    return np.sum(weight[:, np.newaxis] * X, axis=0) / np.sum(weight)

def meanshift_step(X, bandwidth=2.5):
    # raise NotImplementedError('meanshift_step function not implemented!')
    for i in range(X.shape[0]):
        dist = distance(X[i], X)
        weight = gaussian(dist, bandwidth)
        X[i] = update_point(weight, X)
    return X

def meanshift(X):
    BANDWIDTH = 5.0
    for _ in range(20):
        X = meanshift_step(X, BANDWIDTH)
    return X

scale = 0.5    # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('eth.jpg'), scale, channel_axis=-1)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# print(shape)
# print(image_lab.shape)

# Run your mean-shift algorithm
t = time.time()
X = meanshift(image_lab)
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, channel_axis=-1)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result.png', result_image)
