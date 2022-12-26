import cv2
import numpy as np
from collections import deque

# for storage object's center
deque_size = 16
pts = deque(maxlen = deque_size)

# black color range HSV
color_lower = (100, 30, 30)
color_upper = (150, 148, 255)

#capture
cap = cv2.VideoCapture(0)
# capture width, height
cap.set(3, 960)
cap.set(4, 480)

while True:
    success, original_image = cap.read()
    
    if success:
        
        #blur
        blurred = cv2.GaussianBlur(original_image, (11, 11), 0)
        
        #hsv
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        #cv2.imshow("HSV Image", hsv)
        
        # mask for blue
        mask = cv2.inRange(hsv, color_lower, color_upper)
        cv2.imshow("Masked Image", mask)
        
        # erase the mask's nosie
        mask = cv2.erode(mask, None, iterations = 1)
        mask = cv2.dilate(mask, None, iterations = 1)
        cv2.imshow("Masked and Erosion", mask)
        
        #contour
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        center = None
        
        if len(contours) > 0:
            #get the max contour
            c = max(contours, key = cv2.contourArea)
            
            # transform into rectangle
            rect = cv2.minAreaRect(c)
            
            ((x, y), (width, height), rotation) = rect
            
            s = "x: {}, y: {}, width: {}, height: {}, rotation: {}".format(np.round(x), np.round(y), np.round(width), np.round(height), np.round(rotation)) 
            print(s)
            
            #box
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            
            #moment
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            
            # draw contour : yellow
            cv2.drawContours(original_image, [box], 0, (0, 255, 255), 2)
            
            # draw a dot to the center : pink
            cv2.circle(original_image, center, 5, (255, 0, 255), -1)
            
            # write to the screen
            cv2.putText(original_image, s, (25, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)
            
        # deque
        pts.appendleft(center)
        
        for i in range(1, len(pts)):
            if pts[i-1] is None or pts[i] is None: continue
            
            cv2.line(original_image, pts[i-1], pts[i], (0,255,0), 3) # green
            
        cv2.imshow("Blue Tracker", original_image)
            
        
    if cv2.waitKey(1) & 0xFF == ord("q"): break