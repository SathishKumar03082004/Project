import os
import cv2
import numpy as np

def matlab_style_gauss2d(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y)/(2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


# To get a window where center is (x,y) that is of size (N,N)
def get_window(img,x,y,N=25):
    """
    Extracts a small window of input image, around the center (x,y)
    img - input image
    x,y - coordinates of center
    N - size of window (N,N) {should be odd}
    """

    h, w, c = img.shape  # Extracting Image Dimensions
    arm = N // 2  # Arm from center to get window
    window = np.zeros((N, N, c))

    xmin = max(0, x - arm)
    xmax = min(w, x + arm + 1)
    ymin = max(0, y - arm)
    ymax = min(h, y + arm + 1)

    window[arm - (y - ymin):arm + (ymax - y), arm - (x - xmin):arm + (xmax - x)] = img[ymin:ymax, xmin:xmax]

    return window


# Define the folder path
data_folder = 'data'

# Loop through all files in the data folder
for filename in os.listdir(data_folder):
    if filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".jpg"):
        # Construct full file path
        file_path = os.path.join(data_folder, filename)

        # Read the image
        inp_image = cv2.imread(file_path)

        if inp_image is None:
            print(f"Failed to load {file_path}")
            continue

        # Resize the image
        scale_percent = 50  # percent of original size
        width = int(inp_image.shape[1] * scale_percent / 100)
        height = int(inp_image.shape[0] * scale_percent / 100)
        dim = (width, height)
        inp_image = cv2.resize(inp_image, dim, interpolation=cv2.INTER_AREA)

        # Convert to grayscale if the image has multiple channels
        if len(inp_image.shape) > 2:
            res_image = cv2.cvtColor(inp_image, cv2.COLOR_BGR2GRAY)
        else:
            res_image = inp_image

        # Apply threshold
        ret, thresh = cv2.threshold(res_image, 127, 255, cv2.THRESH_BINARY)

        # Show threshold image (optional)
        cv2.imshow(f"Threshold {filename}", thresh)
        cv2.waitKey(0)

        # Find contours
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

        # Draw contours on the original image
        cv2.drawContours(inp_image, contours, -1, (0, 255, 0), 3)

        # Show the image with contours (optional)
        cv2.imshow(f'Contour {filename}', inp_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Normalize the image
        img = np.array(inp_image, dtype='float') / 255

        # Get dimensions of the image
        h, w, c = img.shape

        # Parameters for Gaussian weights
        N = 25
        sig = 8
        minNeighbours = 10

        # Preparing the Gaussian weights for the window
        gaussian_weights = matlab_style_gauss2d((N, N), sig)
        gaussian_weights /= np.max(gaussian_weights)

        print(f"Processed image: {filename}")
