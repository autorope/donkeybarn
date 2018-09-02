import cv2
import numpy as np


# Function that applies Sobel x or y,
# then takes an absolute value and applies a threshold.
def abs_sobel_thresh(img, orient, sobel_kernel, grad_thresh):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    if orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= grad_thresh[0]) & (scaled_sobel <= grad_thresh[1])] = 1
    # Return this mask as your binary_output image
    return binary_output


# Function that applies Sobel x and y,
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel, mgn_thresh):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the magnitude
    abs_sobel = np.sqrt(np.square(np.absolute(sobelx)) + np.square(np.absolute(sobely)))
    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mgn_thresh[0]) & (scaled_sobel <= mgn_thresh[1])] = 1
    # Return this mask as your binary_output image
    return binary_output


# Function that applies Sobel x and y,
# then computes the direction of the gradient
# and applies a threshold.
def dir_thresh(img, sobel_kernel, d_thresh):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    direction = np.arctan2(abs_sobely, abs_sobelx)
    # Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(direction)
    binary_output[(direction > d_thresh[0]) & (direction < d_thresh[1])] = 1
    # Return this mask as your binary_output image
    return binary_output


def LUV_thresh(img, l_thresh):
    img = np.copy(img)
    height, width = img.shape[0], img.shape[1]

    LUV = cv2.cvtColor(img, cv2.COLOR_RGB2LUV).astype(np.float)
    L = LUV[:, :, 0]

    l_binary = np.zeros_like(L)
    l_binary[(L > l_thresh[0]) & (L <= l_thresh[1])] = 1
    return l_binary


def LAB_thresh(img, b_thresh):
    img = np.copy(img)
    height, width = img.shape[0], img.shape[1]

    LAB = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float)
    B = LAB[:, :, 2]

    b_binary = np.zeros_like(B)
    b_binary[(B > b_thresh[0]) & (B <= b_thresh[1])] = 1
    return b_binary


def HLS_thresh(img, s_thresh):
    img = np.copy(img)
    height, width = img.shape[0], img.shape[1]

    HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    S = HLS[:, :, 2]

    s_binary = np.zeros_like(S)
    s_binary[(S > s_thresh[0]) & (S <= s_thresh[1])] = 1
    return s_binary


def HSV_thresh(img, v_thresh):
    img = np.copy(img)
    height, width = img.shape[0], img.shape[1]

    HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    V = HSV[:, :, 2]

    v_binary = np.zeros_like(V)
    v_binary[(V > v_thresh[0]) & (V <= v_thresh[1])] = 1
    return v_binary


def R_G_thresh(img, color_thresh):
    img = np.copy(img)
    height, width = img.shape[0], img.shape[1]

    R = img[:, :, 0]
    G = img[:, :, 1]

    color_combined = np.zeros_like(R)
    r_g_condition = (R > color_thresh) & (G > color_thresh)


def color_thresh_combined(img, s_thresh, l_thresh, v_thresh, b_thresh):
    V_binary = HSV_thresh(img, v_thresh)
    S_binary = HLS_thresh(img, s_thresh)
    L_binary = LUV_thresh(img, l_thresh)

    color_binary = np.zeros_like(V_binary)
    color_binary[(V_binary == 1) & (S_binary == 1) & (L_binary == 1)] = 1
    # color_binary[(V_binary == 1) & (S_binary == 1) & (B_binary == 1) & (L_binary == 1)] = 1

    return color_binary


def combine_thresholds(img, s_thresh, l_thresh, v_thresh, b_thresh,
                       gradx_thresh, grady_thresh, magn_thresh,
                       d_thresh, ksize):
    img = np.copy(img)
    height, width = img.shape[0], img.shape[1]

    binary_x = abs_sobel_thresh(img, 'x', ksize, gradx_thresh)
    binary_y = abs_sobel_thresh(img, 'y', ksize, grady_thresh)
    mag_binary = mag_thresh(img, ksize, magn_thresh)
    dir_binary = dir_thresh(img, ksize, d_thresh)
    color_binary = color_thresh_combined(img, s_thresh, l_thresh, v_thresh, b_thresh)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    binary_output = np.zeros_like(img)
    binary_output[((binary_x == 1) & (binary_y == 1) & (mag_binary == 1)) |
                  (color_binary == 1) | ((mag_binary == 1) & (dir_binary == 1))
                  ] = 1

    # apply the region of interest mask
    mask = np.zeros_like(binary_output)
    region_of_interest_vertices = np.array([[0, height - 1], [width / 2, int(0.4 * height)], [width - 1, height - 1]],
                                           dtype=np.int32)
    cv2.fillPoly(mask, [region_of_interest_vertices], 1)
    thresholded = cv2.bitwise_and(binary_output, mask)

    return thresholded