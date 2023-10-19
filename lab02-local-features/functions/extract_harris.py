import numpy as np

from scipy import signal #for the scipy.signal.convolve2d function
from scipy import ndimage #for the scipy.ndimage.maximum_filter
import cv2

import matplotlib.pyplot as plt  # debug
from matplotlib import colors

from copy import deepcopy

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

    PLOT = False  # Flag to generate report figures
    
    # Generate approx gradient masks and convolve w/ img
    hx = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]]) / 2.  # Naive horizontal gradient mask
    Ix = signal.convolve2d(img, hx, mode="same")

    hy = np.array([[0, -1, 0], [0, 0, 0], [0 , 1, 0]]) / 2.  # Naive vertical gradient mask
    Iy = signal.convolve2d(img, hy, mode="same")

    # Visualise gradient fields
    plt.figure(1)
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

    blurred_Ix = cv2.GaussianBlur(src=Ix, ksize=(5, 5), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    blurred_Iy = cv2.GaussianBlur(src=Iy, ksize=(5, 5), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    # blurred_img = cv2.GaussianBlur(src=img, ksize=(11, 11), sigmaX=10, borderType=cv2.BORDER_REPLICATE)

    fig, (ax1, ax2) = plt.subplots(1,2, num=2, clear=True)
    # plt.imshow(blurred_img, cmap='gray')
    ax1.imshow(blurred_Ix, cmap="gray")
    ax1.set_title("Blurred horizontal intensity gradients")
    ax1.axis("off")

    ax2.imshow(blurred_Iy, cmap="gray")
    ax2.set_title("Blurred vertical intensity gradients")
    ax2.axis("off")
    
    # plt.show()

    # 3. Compute elements of the local auto-correlation matrix "M"
    # TODO: compute the auto-correlation matrix here

    # Perform element-wise multiplications
    Ix2 = blurred_Ix ** 2
    Iy2 = blurred_Iy ** 2
    Ixy = blurred_Ix * blurred_Iy

    # Sum over patch using additional gaussian blur after smoothing gradient
    M11 = cv2.GaussianBlur(src=Ix2, ksize=(5,5), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    M22 = cv2.GaussianBlur(src=Iy2, ksize=(5,5), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    M12 = cv2.GaussianBlur(src=Ixy, ksize=(5,5), sigmaX=sigma, borderType=cv2.BORDER_REPLICATE)
    M21 = M12

    # Sum over patch using standard gaussian blur w/o smoothing gradient
    # M11 = cv2.GaussianBlur(src=Ix**2, ksize=(5,5), sigmaX=SIGMA, borderType=cv2.BORDER_REPLICATE)
    # M22 = cv2.GaussianBlur(src=Iy**2, ksize=(5,5), sigmaX=SIGMA, borderType=cv2.BORDER_REPLICATE)
    # M12 = cv2.GaussianBlur(src=Ix*Iy, ksize=(5,5), sigmaX=SIGMA, borderType=cv2.BORDER_REPLICATE)
    # M21 = M12

    # Simple sum of elements in patch, since gradients already smoothed. Results don't look good however.
    # sum_mask = np.ones((5,5))  # Elements in M are summed over patch
    # M11 = signal.convolve2d(Ix2, sum_mask, mode="same")
    # M22 = signal.convolve2d(Iy2, sum_mask, mode="same")
    # M12 = signal.convolve2d(Ixy, sum_mask, mode="same")
    # M21 = M12

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharex=True, sharey=True, num=3, clear=True)
    fig.suptitle("2nd Moment Gradients")
    im1 = ax1.imshow(M11, cmap="gray")
    ax1.axis("off")
    ax1.set_title("$I_{xx}$")

    im2 = ax2.imshow(M12, cmap="gray")
    ax2.axis("off")
    ax2.set_title("$I_{xy}$")

    im3 = ax3.imshow(M22, cmap="gray")
    ax3.axis("off")
    ax3.set_title("$I_{yy}$")
    

    # 4. Compute Harris response function C
    # TODO: compute the Harris response function C here

    detM = M11 * M22 - M12**2
    trace = M11 + M22
    C = detM - k*trace**2

    # Set response at borders of img to 0
    BORDER = 5
    C[:BORDER, :] = 0  # Top edge
    C[-BORDER:, :] = 0  # Bottom edge
    C[:, :BORDER] = 0  # Left edge
    C[:, -BORDER:] = 0  # Right edge

    print(f"C size: {C.shape}")

    # Delete

    plt.figure(4)
    # Note extreme gradients at edges in partials => extremes at corners. Crop edges for visualisation.
    plt.imshow(C, cmap="gray")
    plt.title("Harris response")
    plt.axis("off")
    plt.colorbar()

    # plt.show()

    # 5. Detection with threshold and non-maximum suppression
    # TODO: detection and find the corners here
    # For the non-maximum suppression, you may refer to scipy.ndimage.maximum_filter to check a 3x3 neighborhood.
    # You may refer to np.where to find coordinates of points that fulfill some condition; Please, pay attention to the order of the coordinates.
    # You may refer to np.stack to stack the coordinates to the correct output format

    # Generate binary image of corner candidates
    thresholded_response = C > thresh
    plt.figure(5)
    plt.imshow(thresholded_response, cmap="gray")
    plt.title("Thresholded Response")
    plt.axis("off")
    plt.colorbar()
    # plt.show()

    # Get locations of sub-threshold harris reponse
    non_candidates = np.stack(np.where(C < thresh), axis=0)  # (357, 2) array containing indices of non corner candidates
    print(f"candidates type {type(non_candidates)}, size: {np.shape(non_candidates)}")
    print(np.array(non_candidates)[:,0])

    # Set non candidate Harris Response to 0 (avoid spurious maxima)
    suppressed = np.zeros_like(C)
    suppressed[C > thresh] = C[C > thresh]

    plt.figure(6)
    plt.imshow(suppressed, cmap="gray")
    plt.title("Suppressed sub-threshold response")
    plt.colorbar()

    # Perform non-max suppresion
    response_max = ndimage.maximum_filter(suppressed, size=(3,3))
    mask = suppressed == response_max  # Mask to select pixels that were the max in neighbourhood
    corners = suppressed * mask

    # corners = ndimage.maximum_filter(thresholded_response, size=(3,3))
    plt.figure(7)
    plt.imshow(corners, cmap="gray")
    plt.title("Corners after non-max suppresion")
    plt.axis("off")
    plt.colorbar()
    # plt.show()


    # binary image of corners
    plt.figure(8)
    plt.imshow(corners > thresh, cmap="gray")
    plt.title("Binary image of corners after non-max suppresion")
    plt.axis("off")
    plt.colorbar()

    if PLOT:
        plt.show()
    else:
        plt.close("all")

    # `corners` should be indices not harris response
    # N.B. opencv images are (m, n) but vis tools use (x, y) convention
    corners = np.stack(np.where(corners > thresh), axis=1)  # (num_corners, 2)
    print("shape:", np.shape(corners))
    corners_swapped = []
    for idxs in corners:
        m, n = idxs
        corners_swapped.append((n,m))

    corners = np.array(corners_swapped)
    return corners, C

