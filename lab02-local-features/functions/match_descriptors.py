import numpy as np

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    # TODO: implement this function please
    distances = np.zeros((desc1.shape[0], desc2.shape[0]))
    for m, keypoint1 in enumerate(desc1):
        # print(f"keypoint shape: {keypoint1.shape}")
        for n, keypoint2 in enumerate(desc2):
            diff = keypoint1 - keypoint2
            # print(f"diff, {np.shape(diff)}")
            # print("transpose diff * diff, ", np.shape(np.transpose(diff) * diff), np.shape(np.dot(diff, diff)))
            distances[m, n] = np.dot(diff, diff)
    print(distances.shape)
    return distances


def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = []
    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1
        # TODO: implement the one-way nearest neighbor matching here
        # You may refer to np.argmin to find the index of the minimum over any axis
        for i, di in enumerate(distances):
            j = np.argmin(di)
            matches.append(np.array([i, j]))
        matches = np.array(matches)
        print(f"one_way shape: {matches.shape}")
    elif method == "mutual":
        # TODO: implement the mutual nearest neighbor matching here
        # You may refer to np.min to find the minimum over any axis

        # Naive solution
        # for i, di in enumerate(distances):
        #     j = np.argmin(di)
        #     reverse_match = np.argmin(distances)
        #     matches.append(np.array([i, j]))
        # matches = np.array(matches)


        min_for_desc1 = np.argmin(distances, axis=1)
        min_for_desc2 = np.argmin(distances, axis=0)

        # print(f"min for desc1 keypoints: {min_for_desc1}")
        # print(f"min for desc2 keypoints: {min_for_desc2}")

        matches = []

        for i, min_j in enumerate(min_for_desc1):
            if i == min_for_desc2[min_j]:
                matches.append([i, min_j])
        
        matches = np.asarray(matches)
        print(f"matches: {matches.shape}")
        

    elif method == "ratio":
        # TODO: implement the ratio test matching here
        # You may use np.partition(distances,2,axis=0)[:,1] to find the second smallest value over a row
        # for i, row in enumerate(distances):
        second_smallest = np.partition(distances,2,axis=0)[:,1]
        print(f"second_smallest: {second_smallest}, shape: {second_smallest.shape}")

        smallest = np.min(distances, axis=1)

        ratio = smallest / second_smallest
        mask = ratio < ratio_thresh
        print(f"mask: {mask}\nmask shape: {mask.shape}")

        min_for_desc1 = np.argmin(distances, axis=1)
        print(f"min for desc1 keypoints shape: {min_for_desc1.shape}")

        unmasked_matches = np.array([(i, j) for i, j in enumerate(min_for_desc1)])
        matches = unmasked_matches[mask]
        print(f"matches: {matches}, matches shape: {matches.shape}")


    else:
        raise NotImplementedError
    return matches

