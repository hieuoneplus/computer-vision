from imutils.perspective import four_point_transform
import cv2
import numpy as np

def rotate(image):
    # Load image, grayscale, Gaussian blur, Otsu's threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # warpedd = cv2.resize(gray, (800, 600))
    # cv2.imshow("gray", warpedd)
    # cv2.waitKey()
    blur = cv2.GaussianBlur(gray, (9,9), 0)
    # warpedd = cv2.resize(blur, (800, 600))
    # cv2.imshow("blur", warpedd)
    # cv2.waitKey()
    # thresh = cv2.adaptiveThreshold(
    #     blur, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # warpedd = cv2.resize(thresh, (800, 600))
    # cv2.imshow("thres", warpedd)
    # cv2.waitKey()
    # Find contours and sort for largest contour
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    displayCnt = None

    for c in cnts:
        # Perform contour approximation
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            displayCnt = approx
            break
    # print(displayCnt)
    if displayCnt is not None:
        # Obtain birds' eye view of image
        warped = four_point_transform(image, displayCnt.reshape(4, 2))
        # warpedd = cv2.resize(warped, (800, 600))
        # cv2.imshow("rotate", warpedd)
        # cv2.waitKey()
        return warped
    else:
        # Return original image if no suitable contour was found
        return image

# image_path = cv2.imread("5.JPG")
# warped = rotate(image_path)
# warped = cv2.resize(warped, (800, 600))
# cv2.imshow("warped", warped)
# cv2.waitKey()