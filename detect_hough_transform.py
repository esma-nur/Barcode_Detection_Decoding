
import os
import cv2
import math
from matplotlib import pyplot as plt
import numpy as np




original_subset_dir = "dataset/Original_Subset"
detection_subset_dir = "dataset/Detection_Subset"
original_subset_image_names = os.listdir(original_subset_dir)
detection_subset_image_names = os.listdir(detection_subset_dir)
line_thickness = 2
plt.style.use("ggplot")
thetas = np.deg2rad(np.arange(-90.0, 90.0))     # Theta range
cos_theta = np.cos(thetas)
sin_theta = np.sin(thetas)
num_thetas = len(thetas)
max_n = 100
sigma = 0.3


def detect_barcode_lines():
    for image_name in original_subset_image_names:
        print(image_name)

        # Read original and ground truth images
        original_img = cv2.imread(original_subset_dir + '/' + image_name)
        detected_img = cv2.imread(detection_subset_dir + '/' + image_name)

        # Obtain edge map of the input image
        edges, plot_input = obtain_edge_map(original_img)

        # Mask edge map with ground truth so only barcode lines will be found
        masked_img = cv2.bitwise_and(detected_img, cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))

        # Utilize Hough transform on edge map
        accumulator, rhos = find_lines(cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY))

        # Transform Hough space to image space
        plot_input = hough_to_image_space(original_img, detected_img, accumulator, rhos, plot_input)

        # Find contours in the edge map
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)



        # Plot result
        plt.imshow(cv2.cvtColor(plot_input, cv2.COLOR_BGR2RGB))
        plt.title(''), plt.xticks([]), plt.yticks([])
        plt.show()
        
        # Save the plot_input to a file
        cv2.imwrite('barcode_detect/{}.png'.format(image_name), plot_input)

        # Close the plot window
        plt.close()


       
def obtain_edge_map(img):
    # Blur the image to reduce noise
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray_img, 7)

    # Apply Canny edge detection to image
    median = np.median(img)
    lower_threshold = int(max(0, (1.0 - sigma) * median))
    upper_threshold = int(min(255, (1.0 + sigma) * median))
    edges = cv2.Canny(blurred, lower_threshold, upper_threshold)

    # Concat original and edge image for plot
    plot_input = np.concatenate((img, cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)), axis=1)
    return edges, plot_input

    


def find_lines(img):
    width = img.shape[0]
    height = img.shape[1]

    # Diagonal length of the image
    diagonal_length = int(np.ceil(np.sqrt(width * width + height * height)))

    # Rho ranges from -diagonal_length to +diagonal_length of the image
    rhos = np.linspace(-diagonal_length, diagonal_length, diagonal_length * 2)

    # Hough transform vote accumulator: 2 * diagonal_length rows, num_thetas columns
    accumulator = np.zeros((2 * diagonal_length, num_thetas), dtype=np.uint64)

    # Get (row, col) indexes to edges indices that are non-zero
    y_idx, x_idx = np.nonzero(img)

    # Vote in accumulator
    for i in range(len(x_idx)):
        x = x_idx[i]
        y = y_idx[i]

        # Calculate current rhos
        curr_rhos = np.add(np.array(cos_theta) * x, np.array(sin_theta) * y)

        for t_idx in range(num_thetas):
            # Increment vote of that theta and closest rho by one
            accumulator[int(round(curr_rhos[t_idx]) + diagonal_length), t_idx] += 1
    return accumulator, rhos


def hough_to_image_space(original_img, detected_img, accumulator, rhos, plot_input):
    # Get threshold
    threshold = get_avg_threshold(accumulator)

    # Get indices of the most clear lines according to threshold
    y_idx, x_idx = np.where(accumulator >= threshold)

    for i in range(len(y_idx)):
        # Get flatten index to find rho and theta
        flatten_idx = get_flatten_idx(accumulator.shape[1], y_idx[i], x_idx[i])

        # Find rho and theta parameter of current line
        rho = rhos[int(round(flatten_idx / accumulator.shape[1]))]
        theta = thetas[flatten_idx % accumulator.shape[1]]

        # Convert Hough space to image space
        cos = math.cos(theta)
        sin = math.sin(theta)
        x = cos * rho
        y = sin * rho
        from_point = (int(x + 1000 * (-sin)), int(y + 1000 * cos))
        to_point = (int(x - 1000 * (-sin)), int(y - 1000 * cos))

        # Draw lines on original and detected images
        cv2.line(original_img, from_point, to_point, (0, 0, 255), line_thickness)
        cv2.line(detected_img, from_point, to_point, (0, 0, 255), line_thickness)
        
        

    plot_input = np.concatenate((plot_input, original_img), axis=1)
    plot_input = np.concatenate((plot_input, detected_img), axis=1)
    return plot_input

    #cv2.imshow("original_img",original_img)
    #cv2.imshow("detected_img",detected_img)
    #cv2.waitKey(0)


def get_n_max_idx(arr):
    # Find n number of maximum numbers in arr
    indices = np.argpartition(arr, arr.size - max_n, axis=None)[-max_n:]
    return np.column_stack(np.unravel_index(indices, arr.shape))


def get_avg_threshold(arr):
    # Get max_n number of maximum numbers in arr
    n_max_idx = get_n_max_idx(arr)

    # Calculate average of max_n numbers
    sum = 0.0
    for i in range(len(n_max_idx)):
        sum += arr[n_max_idx[i][0]][n_max_idx[i][1]]
    return sum/len(n_max_idx)


def get_flatten_idx(total_cols, row, col):
    return (row * total_cols) + col


if __name__ == '__main__':
    detect_barcode_lines()
    