import os
import cv2
import math
import time
import random
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt

class FindPainting:
  debug = False


  def __init__(self, debug=False):
    self.debug = debug


  def FindPainting(self, img_original):

    def mean_shift_seg(img,s_radius, c_radius, pyramid_level):
      img_ms_seg = cv2.pyrMeanShiftFiltering(img, s_radius, c_radius, pyramid_level)
      return img_ms_seg

    # Create a mask of the largest segment (wall)
    def largest_segment(img,color):
      assert img is not None
      cl = img.copy()
      mask = np.zeros((img.shape[0]+2, img.shape[1]+2), dtype=np.uint8)
      wallColor = None
      largest_segment = 0
      for y in range(img.shape[0]):
          for x in range(img.shape[1]):
              if mask[y+1, x+1] == 0:
                  newVal = (np.random.randint(0,256), np.random.randint(0,256), np.random.randint(0,256))
                  retval, image, mask, rect = cv2.floodFill(cl, mask, (x, y), newVal, loDiff=color, upDiff=color, flags=4)

                  segment_size = rect[2] * rect[3]
                  if segment_size > largest_segment:
                      largest_segment = segment_size
                      wallColor = newVal
      return cv2.inRange(cl, wallColor, wallColor)

    # Dilate the mask to remove noise
    def dilate(img, size):
      kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size,size))
      dilated = cv2.dilate(img, kernel)
      return dilated

    # Invert the mask
    def invert(img):
      inverted = cv2.bitwise_not(img)
      return inverted

    # Refine the list of components found
    def couldBePainting(img, bounder, contour, width, height, area_percentage):
        # Check that the rect is smaller than the entire image and bigger than a certain size
        if bounder[2] * bounder[3] < img.shape[0] * img.shape[1] and bounder[2] * bounder[3] > width * height:
            # Extra to remove floors when programming
            if cv2.contourArea(contour) > bounder[2] * bounder[3] * area_percentage:
                return True
        return False

    def getPossiblePaintingContours(img, contours, min_width=150, min_height=150, min_area_percentage=.55):
        painting_contours = []
        if contours:
            for contour in contours:
                bounder = cv2.boundingRect(contour)
                if couldBePainting(img, bounder, contour, min_width, min_height, min_area_percentage):
                    painting_contours.append(contour)
        return painting_contours

    # Erode the image to get rid of things attached to the frame
    def erode(img, se_size, shape=cv2.MORPH_RECT):
      return cv2.erode(img, cv2.getStructuringElement(shape, (se_size, se_size)))

    # Apply median filter to the eroded mask
    def median_filter(img, blur_size):
        return cv2.medianBlur(img, blur_size)

    def cannyEdgeDetection(img):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        otsu_thresh_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        high_thresh_val = otsu_thresh_val
        lower_thresh_val = otsu_thresh_val * 0.5
        edges = cv2.Canny(gray, lower_thresh_val, high_thresh_val)
        return edges

    def partition(lines, labels, areEqual):
        numGroups = 0
        for i, line1 in enumerate(lines):
            groupAssigned = False
            for j in range(numGroups):
                line2 = lines[int(labels[j])]
                if areEqual:
                    labels[i] = j
                    groupAssigned = True
                    break
            if not groupAssigned:
                labels[i] = numGroups
                numGroups += 1
        return numGroups, labels

    def linesAreEqual(l1, l2):
        length1 = np.sqrt((l1[2] - l1[0])**2 + (l1[3] - l1[1])**2)
        length2 = np.sqrt((l2[2] - l2[0])**2 + (l2[3] - l2[1])**2)
        product = (l1[2] - l1[0])*(l2[2] - l2[0]) + (l1[3] - l1[1])*(l2[3] - l2[1])
        if abs(product / (length1 * length2)) < math.cos(np.pi/30):
            return False
        mx1 = (l1[0] + l1[2]) * 0.5
        mx2 = (l2[0] + l2[2]) * 0.5
        my1 = (l1[1] + l1[3]) * 0.5
        my2 = (l2[1] + l2[3]) * 0.5
        dist = np.sqrt(((mx1 - mx2)**2) + ((my1 - my2)**2))
        if dist > max(length1, length2) * 0.5:
            return False
        return True

    # Returns the lines found from performing HoughLinesP on the image
    def geoffHoughLines(img, rho, theta, hough_threshold, min_line_length, min_line_gap):
        lines = cv2.HoughLinesP(img, rho, theta, hough_threshold, minLineLength=min_line_length, maxLineGap=min_line_gap)
        return lines

    # Groups similar lines in angle and position together and returns this new vector of lines
    def groupHoughLines(lines):
        # Find number of groups of lines that are similar
        labels = np.zeros(len(lines))
        equal = []
        numberOfLines = 0
        for i in range(len(lines)-1):
          equal.append(linesAreEqual(lines[i][0],lines[i+1][0]))
        for same in equal:
          numberOfLines, labels = partition(lines, labels, same)
        groupedLines = []
        #Group together all lines from the same group
        for j in range(numberOfLines):
          tlx = 2147483647; tly = 2147483647; brx = -1; bry = -1;
          for k in range(len(lines)):
            if int(labels[k]) == j:
              x1, y1, x2, y2 = lines[k][0]
              groupedLines.append([x1, y1, x2, y2])
        return groupedLines

    def extendLinesAcrossImage(img, lines, color):
        res = img.copy()
        for line in lines:
            hoek = math.atan2(line[1] - line[3], line[0] - line[2])
            hoek = hoek * 180 / math.pi
            if abs(hoek) > 45:
              length = max(img.shape[0], img.shape[1])
              P1 = (line[0], line[1])
              P2 = (int(round(P1[0] + length * math.cos(hoek * math.pi / 180.0))), int(round(P1[1] + length * math.sin(hoek * math.pi / 180.0))))
              P3 = (int(round(P1[0] - length * math.cos(hoek * math.pi / 180.0))), int(round(P1[1] - length * math.sin(hoek * math.pi / 180.0))))
              cv2.line(res, P3, P2, color, 10, cv2.LINE_8 )
        return res

    def findLargestContour(sudoku):
      gray_image = cv2.cvtColor(sudoku, cv2.COLOR_BGR2GRAY)
      contours, hierarchy = cv2.findContours(cv2.bitwise_not(gray_image), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
      
      # If the mask does not look like we expect (like a sudoku puzzle) then give up at this point :(
      if len(contours) < 9:
          return None
      
      # Find the largest contour from the sudoku version of the image
      max_contour = 0
      for j in range(len(contours)):
          if len(contours[j]) > len(contours[max_contour]):
              max_contour = j
      
      return contours[max_contour]

    def harrisCornerDetection(img, max_corners, quality, minimum_distance):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, max_corners, quality, minimum_distance)
        corners = corners.reshape(-1, 2)
        return corners.astype(int)

    def order_corners(corners):
        corners = corners.squeeze()
        x0 = corners[0]
        x1 = corners[0]
        x2 = corners[0]
        x3 = corners[0]
        sorted = []
        k = 0
        j = 0
        for index, i in enumerate(corners):
          if i[0] < x0[0] or i[1] < x0[1]:
            if i[0]+i[1] < x0[0] + x0[1]:
              x0 = i
              k = index
          if i[0] > x2[0] or i[1] > x2[1]:
            if i[0]+i[1] > x1[0] + x1[1]:
              x2 = i
              j = index
        for index, i in enumerate(corners):
          if i[0] > x1[0] or i[1] < x1[1]:
            if index != k and index != j:
              x1 = i
          if i[0] < x3[0] or i[1] > x3[1]:
            if index != k and index != j:
              x3 = i
        sorted = []
        sorted.append(x0)
        sorted.append(x1)
        sorted.append(x2)
        sorted.append(x3)
        return sorted
  # Start --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Variabelen------------------------------------------
    # Mean-shift
    spatial_radius = 7
    color_radius = 13
    maximum_pyramid_level = 1

    #largest segment
    color_difference = (2, 2, 2, 0)  

    # Dilate
    structuring_element_size = 18

    # Seperate image
    offset = 15

    # Erosion
    erosion_structuring_element_size = 60

    # Median filter
    blur_size = 31

    # Harris corner detection
    corner_quality = 0.001
    minimum_distance = 20
    #--------------------------------------------------------
    startTime = time.time()

    # Perform Mean-Shift Segmentation on the image
    image_original = img_original.copy()
    ms_seg = mean_shift_seg(image_original, spatial_radius, color_radius, maximum_pyramid_level)

    # Create a mask of the largest segment
    wall_mask = largest_segment(ms_seg,color_difference)

    # Dilate the mask to remove noise
    dilated_mask = dilate(wall_mask, structuring_element_size)

    # Invert the mask
    inverted_mask = invert(dilated_mask)

    # Perform Connected Components Analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted_mask)
    image_labels = cv2.convertScaleAbs(labels)

    # Refine the list of components found
    contours, hierarchy = cv2.findContours(image_labels, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    frame_contours = getPossiblePaintingContours(image_labels, contours)
    cl = img_original.copy()
    # create a mask for the area inside the contours
    mask = np.zeros_like(cl)
    cv2.drawContours(mask, frame_contours, -1, color=(255, 255, 255), thickness=cv2.FILLED)
    # fill the area outside the contours with a different color
    cl[np.where(mask == 0)[0], np.where(mask == 0)[1], :] = np.array([(0, 0, 0)])
    # draw filled contours on the image
    for i in range(len(frame_contours)):
        color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        cv2.drawContours(cl, frame_contours, i, color, cv2.FILLED)

    # Get the seperate images
    images = []
    images_original = []
    rows, cols = (len(frame_contours), 2)
    start_pt = [[0 for x in range(cols)] for z in range(rows)]
    for i,contour in enumerate(frame_contours):
      off_x, off_y, off_w, off_h = cv2.boundingRect(contour)
      start_pt[i][0] = off_x
      start_pt[i][1] = off_y
      cropped_img = cl[off_y:off_y+off_h, off_x:off_x+off_w]
      images.append(cropped_img)
      cropped_img = image_original[off_y:off_y+off_h, off_x:off_x+off_w]
      images_original.append(cropped_img)

    # Erode the images to get rid of things attached to the frame
    err = []
    for image in images:
      err.append(erode(image, erosion_structuring_element_size))  

    # Apply median filter to the eroded mask
    med = []
    for image in err:
      med.append(median_filter(image, blur_size))  

    # Apply canny edge to the median filter mask
    edges = []
    indexen = []
    for i,image in enumerate(med):
      img_edges = cannyEdgeDetection(image)
      all_zeros = not np.any(img_edges)
      if all_zeros == False:
        edges.append(img_edges)
        indexen.append(i)

    # Groups similar lines in angle and position together and returns this new vector of lines
    grouped_lines_per_img = []
    Lines_all = []
    remove_index = 0
    for i,index in enumerate(indexen):
      painting_ratio = int(max(edges[i].shape[0], edges[i].shape[1])*0.1)
      lines = geoffHoughLines(edges[i], 1, np.pi/180, 0, painting_ratio*1.25, painting_ratio)
      Lines_all.append(lines)
    for i, lines in enumerate(Lines_all):
      all_zeros = not np.any(lines)
      if all_zeros == False:
        grouped_lines = groupHoughLines(lines)
        grouped_lines_per_img.append(grouped_lines)
      else:
        del indexen[i - remove_index]
        remove_index += 1

    img_lines = []
    for i, index in enumerate(indexen):
      mask = np.zeros_like(images[index])
      for j in range(len(grouped_lines_per_img[i])):
        x1, y1, x2, y2 = grouped_lines_per_img[i][j]
        cv2.line(mask, (x1, y1), (x2, y2), (255, 255, 255), 5)
      img_lines.append(mask)  

    #Returns a version of the image where the lines have been drawn on it in 'color' (extended to reach across the entire image
    sudoku_all = []
    for i,index in enumerate(indexen):
      sudoku = np.zeros_like(images[index])
      sudoku = extendLinesAcrossImage(sudoku, grouped_lines_per_img[i], (255, 255, 255))
      sudoku_all.append(sudoku) 

    # Find the largest contour from the sudoku version of the image
    largestContour = []
    remove_index = 0
    contour_all = []
    for i,index in enumerate(indexen):
      Contour = findLargestContour(sudoku_all[i])
      contour_all.append(Contour)
    for i, contour in enumerate(contour_all):
      if contour is not None:
        largestContour.append(contour)
      else:
        del indexen[i - remove_index]
        remove_index += 1
    img_largestcontour = []
    img_largestcontour_orig = []
    img_original_copy = img_original.copy()
    for i, index in enumerate(indexen):
      # create a mask for the area inside the contours
      mask = np.zeros_like(images[index], np.uint8)
      cv2.drawContours(mask, largestContour[i], -1, color=(255, 0, 0), thickness=cv2.FILLED)    
      cv2.drawContours(img_original_copy, largestContour[i], -1, color=(255, 0, 0), thickness=cv2.FILLED)
      img_largestcontour.append(mask)

    # Find the corners of the painting and plot it on the image
    corners_all = []
    for img in img_largestcontour:
      corners = harrisCornerDetection(img, 4, corner_quality, minimum_distance)
      corners_all.append(corners)

    sorted_corners_all = []
    for i in range(len(corners_all)):
      sorted_corners = order_corners(corners_all[i])
      sorted_corners = np.array(sorted_corners)
      sorted_corners_all.append(sorted_corners)

    img_corners1 = []
    rows, cols = (len(corners_all), 2)
    coord = [[[0 for x in range(cols)] for z in range(4)] for z in range(rows)]

    for i, img in enumerate(img_largestcontour):
      # create a mask for the area inside the contours
      mask = images_original[indexen[i]]
      for j,corner in enumerate(corners_all[i]):
        x, y = corner
        coord[i][j][0] += x
        coord[i][j][1] += y
        #cv2.circle(mask, (x,y), int(0.01 * max(img.shape[0], img.shape[1])), (255, 0, 0), -1)
      img_corners1.append(mask)

    if self.debug: print('Duration: ', time.time() - startTime, 's')

    return img_corners1

