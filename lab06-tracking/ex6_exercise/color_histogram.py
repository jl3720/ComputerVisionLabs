import cv2
import numpy as np

def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin):
    """
    This function should calculate the normalized histogram of RGB 
    colors occurring within the bounding box defined by (xmin, xmax) (ymin, ymax) 
    within the current video frame. The histogram is obtained by binning each 
    color channel (R,G,B) into hist bin bins
    """
    # Clip the bounding box coordinates to be within the frame
    xmin = np.clip(xmin, 0, frame.shape[1] - 1)
    xmax = np.clip(xmax, 0, frame.shape[1] - 1)
    ymin = np.clip(ymin, 0, frame.shape[0] - 1)
    ymax = np.clip(ymax, 0, frame.shape[0] - 1)
    # print(f"frame shape: {frame.shape}")
    # Extract the region of interest from the frame
    roi = frame[ymin:ymax, xmin:xmax]

    # Convert the ROI to the RGB color space
    # roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi_rgb = roi  # already RGB

    # Split the ROI into individual color channels
    r, g, b = cv2.split(roi_rgb)

    # Calculate the histogram for each color channel
    hist_r, _ = np.histogram(r, bins=hist_bin, range=(0, 256))
    hist_g, _ = np.histogram(g, bins=hist_bin, range=(0, 256))
    hist_b, _ = np.histogram(b, bins=hist_bin, range=(0, 256))

    # Normalize the histograms
    hist_r = hist_r / np.sum(hist_r)
    hist_g = hist_g / np.sum(hist_g)
    hist_b = hist_b / np.sum(hist_b)

    # Concatenate the histograms of all color channels
    hist = np.concatenate((hist_r, hist_g, hist_b))

    return hist.flatten()
    
    