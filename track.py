#! /usr/bin/env python
import os
import sys
import csv
import cv2
import glob
import numpy as np
from skimage.measure import LineModel, ransac


# REQUIRES: (numpy array, condition)
# RETURNS : returns splitted array by given condition
def split(arr, cond):
    return [ arr[cond, :],  arr[~cond, :] ]

# REQUIRES: two points on line
# RETURNS : returns lines's intercept with camera horizion 
def x_intercept(line_xi, line_xj, line_yi, line_yj):
    delta = (line_yj - line_yi)/(line_xj - line_xi)   # Slope of RANSAC determined Line y= delta*x + c
    c = (line_yi-line_xi*delta)                       # Constant of RANSAC determined Line
    x = (480 - c)/delta                                # Calculated X-intercept

    if (x<0 or x>640):                                #Return -1 if predicted Lane dosnt cross bottom of image
        x = -1
    return int(x)


if __name__ == "__main__":
    
    width_img, height_img = 640, 480   # Target Image
    max_roi = 0.85                     # Max ROI cofficient to calculate size of ROI
    min_roi = 0.55                     # Min ROI cofficient to calculate size of ROI
    Frame_split = 290                  # Splitting frame vertically to calculate Left and Right lane seperately (Found Experimentally,depends on Camera Location)
    yellow = (0,255,255)               # yellow
    blue = (255, 0 , 0)                # blue
    green = (0, 255, 0)                # green
    roi2img_t = 264                    # Y-axis Transform from ROI to img 

    # Line data
    cv2.namedWindow('Lane Markers')
    imgs = glob.glob("images/*.png")
    
    intercepts = []

    # X-axis seed for RANSAC Y-axis
    line_x = np.arange(0,640)

    #debugging 
    #black = np.zeros((int(height_img*0.85)-int(height_img*0.55), width_img, 3), np.uint8)

    for fname in imgs:
        # Load image and prepare output image
        img = cv2.imread(fname)

        # Resizing image thus reducing computation
        # also, making it easier to use on a laptop screen
        img = cv2.resize(img, (width_img, height_img)) 
        
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Extracting ROI (Lane section)
        roi = frame[int(height_img*min_roi):int(height_img*max_roi), 0:width_img ]

        # Extratring edges 
        edge = cv2.Canny(roi, 40, 250)

        # Extracting line segments using Probablistic Hough Lines
        line = cv2.HoughLinesP(edge, 1, np.pi/180, 80, 30, 10)


        #debugging
        '''
        try: 
            for x1,y1,x2,y2 in line[0]:
                cv2.circle(black,(x1, y1), 2, (255, 0, 0), -1 )
                cv2.circle(black,(x2, y2), 2, (255, 0, 0), -1 )
                ang = int(np.rad2deg((np.arctan2((y2, y1), (x2, x1)))[0]))
                print ang
                if ( (ang>30 and ang<80) or (ang>100 and ang<150)   :
                
                #cv2.line(roi,(x1, y1), (x2,y2), (0,255,0),3 )
                #cv2.line(black,(x1, y1), (x2,y2), (255,255,255),2 )
        except (AttributeError):
            print "Attribute Error"
        '''

        # RANSAC line calculation
        try:
            # Copying line data from P Hough Line
            line_ransac = line[0]

            # Converting (N,4) -> (N,2)
            line_ransac = np.hsplit(line_ransac, 2)
            a = np.array(line_ransac[0])
            b = np.array(line_ransac[1])
            line_ransac = np.concatenate((a, b), axis = 0)

            # Sorting P Hough Line data (increasing X coordiate)    
            line_ransac = line_ransac[line_ransac[:,0].argsort()]
            
            # Splitting P Hough Line data between Left and Right lane data
            line_ransac = split(line_ransac, line_ransac[:,0]<290)

            # Side_l left lane data points
            side_l = np.array(line_ransac[0])

            # Side_r right lane data points
            side_r = np.array(line_ransac[1])

            #RANSAC line fitting
            model_l, inline_l = ransac(side_l, LineModel, min_samples=2,
                                    residual_threshold=1, max_trials=500)
            model_r, inline_r = ransac(side_r, LineModel, min_samples=2,
                                    residual_threshold=1, max_trials=500)
            
            outliers_l = inline_l == False
            outliers_r = inline_r == False

            # Calculated Left Lane
            line_y_l = model_l.predict_y(line_x)

            # Calculated Right Lane
            line_y_r = model_r.predict_y(line_x)

        except(ValueError, TypeError):
            print "Skipping frame, RANSAC unsucessful"
        
        # Calcuating Left image bottom intercept
        intercept_l = x_intercept(line_x[0], line_x[-1],  roi2img_t + line_y_l[0],  roi2img_t + line_y_l[-1])
        # Calcuating Right imgage bottom intercept
        intercept_r = x_intercept(line_x[0], line_x[-1],  roi2img_t + line_y_r[0],  roi2img_t + line_y_r[-1])

        # Draw sample lane markers
        (height, width) = img.shape[:2]
        left_x = intercept_l
        right_x = intercept_r

        # Displaying Lines and Circle 
        cv2.circle(img,(int(intercept_l), 480), 10, blue, -1 )
        cv2.circle(img,(int(intercept_r), 480), 10, blue, -1 )
        cv2.line(img, (line_x[0], roi2img_t + int(line_y_l[0])), (line_x[-1],  roi2img_t + int(line_y_l[-1])), yellow, 2)
        cv2.line(img, (line_x[0], roi2img_t + int(line_y_r[0])), (line_x[-1],  roi2img_t + int(line_y_r[-1])), yellow, 2)
        cv2.line(img, (0,480), (width, 480), green, 2)
        
        if (left_x == -1):
            left_x = "None"
        if (right_x == -1):
            right_x = "None"

        # Sample intercepts
        intercepts.append((os.path.basename(fname), left_x, right_x))

        # debugging
        #cv2.imshow("Edge", edge)
        #cv2.imshow('Hough Transform', roi)
        #cv2.imshow('points', black)
        
        # Show Image
        cv2.imshow('Lane Markers', img)
        

        key = cv2.waitKey(4)
        if key == 27:
            sys.exit(0)
        if key == ord(' '):
            cv2.waitKey(0)
                
    # CSV output
    with open('intercepts.csv', 'w') as f:
        writer = csv.writer(f)    
        writer.writerows(intercepts)
        
    cv2.destroyAllWindows();
    	
