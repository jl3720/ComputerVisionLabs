import numpy as np

from scipy import signal #for the scipy.signal.convolve2d function
from scipy import ndimage #for the scipy.ndimage.maximum_filter
import cv2

import matplotlib.pyplot as plt  # debug
from matplotlib import colors

# Harris corner detector
def extract_harris(img, sigma = 1.0, k = 0.05, thresh = 1e-5):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''
    # Convert to float
    img = img.astype(float) / 255.0

    # 1. Compute image gradients in x and y direction
    # TODO: implement the computation of the image gradients Ix and Iy here.
    # You may refer to scipy.signal.convolve2d for the convolution.
    # Do not forget to use the mode "same" to keep the image size unchanged.
    
    # Generate approx gradient masks and convolve w/ img
    hx = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]]) / 2.  # Naive horizontal gradient mask
    Ix = signal.convolve2d(img, hx, mode="same")

    hy = np.array([[0, -1, 0], [0, 0, 0], [0 , 1, 0]]) / 2.  # Naive vertical gradient mask
    Iy = signal.convolve2d(img, hy, mode="same")

    # Visualise gradient fields
    plt.subplot(1,2,1)
    plt.imshow(Ix, cmap="gray")
    plt.title("Horizontal intensity gradients")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(Iy, cmap="gray")
    plt.title("Vertical intensity gradients")
    plt.axis("off")

    # plt.show()
    
    # 2. Blur the computed gradients
    # TODO: compute the blurred image gradients
    # You may refer to cv2.GaussianBlur for the gaussian filtering (border_type=cv2.BORDER_REPLICATE)

    SIGMA = 1  # scale of blurring

    blurred_Ix = cv2.GaussianBlur(src=Ix, ksize=(5, 5), sigmaX=SIGMA, borderType=cv2.BORDER_REPLICATE)
    blurred_Iy = cv2.GaussianBlur(src=Iy, ksize=(5, 5), sigmaX=SIGMA, borderType=cv2.BORDER_REPLICATE)
    # blurred_img = cv2.GaussianBlur(src=img, ksize=(11, 11), sigmaX=10, borderType=cv2.BORDER_REPLICATE)

    plt.figure()
    # plt.imshow(blurred_img, cmap='gray')
    plt.subplot(1,2,1)
    plt.imshow(blurred_Ix, cmap="gray")
    plt.title("Blurred horizontal intensity gradients")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(blurred_Iy, cmap="gray")
    plt.title("Blurred vertical intensity gradients")
    plt.axis("off")
    
    # plt.show()

    # 3. Compute elements of the local auto-correlation matrix "M"
    # TODO: compute the auto-correlation matrix here

    # Perform element-wise multiplications
    Ix2 = blurred_Ix ** 2
    Iy2 = blurred_Iy ** 2
    Ixy = blurred_Ix * blurred_Iy

    # Sum over patch using additional gaussian blur (double blurring)
    # M11 = cv2.GaussianBlur(src=Ix2, ksize=(5,5), sigmaX=SIGMA, borderType=cv2.BORDER_REPLICATE)
    # M22 = cv2.GaussianBlur(src=Iy2, ksize=(5,5), sigmaX=SIGMA, borderType=cv2.BORDER_REPLICATE)
    # M12 = cv2.GaussianBlur(src=Ixy, ksize=(5,5), sigmaX=SIGMA, borderType=cv2.BORDER_REPLICATE)
    # M21 = M12

    # Sum over patch using standard gaussian blur
    M11 = cv2.GaussianBlur(src=Ix**2, ksize=(5,5), sigmaX=SIGMA, borderType=cv2.BORDER_REPLICATE)
    M22 = cv2.GaussianBlur(src=Iy**2, ksize=(5,5), sigmaX=SIGMA, borderType=cv2.BORDER_REPLICATE)
    M12 = cv2.GaussianBlur(src=Ix*Iy, ksize=(5,5), sigmaX=SIGMA, borderType=cv2.BORDER_REPLICATE)
    M21 = M12

    # Simple sum of elements in patch, since gradients already smoothed. Results don't look good however.
    # sum_mask = np.ones((5,5))  # Elements in M are summed over patch
    # M11 = signal.convolve2d(Ix2, sum_mask, mode="same")
    # M22 = signal.convolve2d(Iy2, sum_mask, mode="same")
    # M12 = signal.convolve2d(Ixy, sum_mask, mode="same")
    # M21 = M12

    plt.figure()
    plt.axis("off")
    plt.subplot(1,3,1)
    plt.imshow(Ix2, cmap="gray")
    plt.title("2nd Moment Gradients")

    plt.subplot(1,3,2)
    plt.imshow(M11, cmap="gray")
    

    # 4. Compute Harris response function C
    # TODO: compute the Harris response function C here

    detM = M11 * M22 - M12**2
    trace = M11 + M22
    C = detM - k*trace**2

    print(f"C size: {C.shape}")

    plt.figure()
    plt.imshow(C[10:246, 10:246], cmap="gray", norm="linear")  # Note extreme gradients at edges in partials => extremes at corners
    plt.title("Harris response")
    plt.axis("off")
    plt.colorbar()

    plt.show()

    # 5. Detection with threshold and non-maximum suppression
    # TODO: detection and find the corners here
    # For the non-maximum suppression, you may refer to scipy.ndimage.maximum_filter to check a 3x3 neighborhood.
    # You may refer to np.where to find coordinates of points that fulfill some condition; Please, pay attention to the order of the coordinates.
    # You may refer to np.stack to stack the coordinates to the correct output format
    raise NotImplementedError
    return corners, C
