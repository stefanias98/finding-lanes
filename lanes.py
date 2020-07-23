import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

global_slope = 0
global_intercept = 0
global_left_line = np.zeros((1,4), dtype=int)
global_center_line = np.zeros((1,4), dtype=int)
global_right_line = np.zeros((1,4), dtype=int)

def canny(image, minVal, maxVal):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    darker_gray = adjust_gamma(gray, 0.5)
    blur = cv2.GaussianBlur(darker_gray, (5,5), 0)
    canny = cv2.Canny(blur, minVal, maxVal)

    return canny

def reg_of_int(image):
    height, width = image.shape
    polygons = np.array([[(0, 2*height/3), (width, 2*height/3), (width, height), (0, height)]], dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    i = 1
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            if i == 1:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
            elif i == 2:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
            elif i == 3:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
            i += 1
    return line_image

def avg_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    center_fit = []
    global global_left_line, global_center_line, global_right_line
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0 and intercept < image.shape[0]:
            left_fit.append((slope, intercept))
            # print("left_fit", left_fit)
        elif slope < 0 and intercept > image.shape[0]:
            center_fit.append((slope, intercept))
            # print("center_fit", center_fit)
        else:
            right_fit.append((slope, intercept))
            # print("right_fit", right_fit)
    if left_fit:
        left_fit_average = np.average(left_fit, axis = 0)
        left_line = make_coordinates(image, left_fit_average)
        global_left_line = left_line
    if center_fit:
        center_fit_average = np.average(center_fit, axis = 0)
        center_line = make_coordinates(image, center_fit_average)
        global_center_line = center_line
    if right_fit:
        right_fit_average = np.average(right_fit, axis = 0)
        right_line = make_coordinates(image, right_fit_average)
        global_right_line = right_line

    print(global_left_line, global_right_line, global_center_line)
    return np.array([global_left_line, global_right_line, global_center_line])

def make_coordinates(image, line_parameters):
    try:
        slope, intercept = line_parameters
        global_slope, global_intercept = line_parameters
    except TypeError:
        slope, intercept = global_slope, global_intercept

    y1 = image.shape[0]
    y2 = int(y1 * 3/5)
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)

    return np.array([x1, y1, x2, y2])

def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)
# image = cv2.imread('image3.jpg')
# lane_image = np.copy(image)
# canny_image = canny(lane_image)
# cropped_image = reg_of_int(canny_image)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=70, maxLineGap=5)
# averaged_lines = avg_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, averaged_lines)
# combined_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
#
# cv2.imshow('result', combined_image)
# cv2.waitKey(0)

cap = cv2.VideoCapture("testData/"+str(sys.argv[1]))

while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame, 50, 150)
    cropped_image = reg_of_int(canny_image)
    lines = cv2.HoughLinesP(cropped_image, cv2.HOUGH_PROBABILISTIC, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = avg_slope_intercept(frame, lines)
    try:
        line_image = display_lines(frame, averaged_lines)
    except OverflowError:
        print("Overflow bato")
    combined_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('result', combined_image)

    if cv2.waitKey(40) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

#_, frame2 = cap2.read()
#canny_image2 = canny(frame2, 50, 150)
#cropped_image2 = reg_of_int(canny_image2)
#lines2 = cv2.HoughLinesP(cropped_image2, cv2.HOUGH_PROBABILISTIC, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
#averaged_lines2 = avg_slope_intercept(frame2, lines2)
#try:
#    line_image2 = display_lines(frame2, averaged_lines2)
#except OverflowError:
#    print("Overflow bato")
#combined_image2 = cv2.addWeighted(frame2, 0.8, line_image2, 1, 1)
#cv2.imshow('result 2', combined_image2)
#
#_, frame3 = cap3.read()
#canny_image3 = canny(frame3, 50, 150)
#cropped_image3 = reg_of_int(canny_image3)
#lines3 = cv2.HoughLinesP(cropped_image3, cv2.HOUGH_PROBABILISTIC, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
#averaged_lines3 = avg_slope_intercept(frame3, lines3)
#try:
#    line_image3 = display_lines(frame3, averaged_lines3)
#except OverflowError:
#    print("Overflow bato")
#combined_image3 = cv2.addWeighted(frame3, 0.8, line_image3, 1, 1)
#cv2.imshow('result 3', combined_image3)
