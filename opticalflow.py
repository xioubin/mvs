import os
import time
import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.cluster import KMeans

# This function allows to calculate optical flow trajectories (Don't remember where I actually found the source code)
# The code also allows to specify step value. The greater the value the more sparse the calculation and visualisation
def calc_angl_n_transl(img, flow, step=8):
    
    '''
    input:
        - img - numpy array - image
        - flow - numpy array - optical flow
        - step - int - measurement of sparsity
    output:
        - angles - numpy array - array of angles of optical flow lines to the x-axis
        - translation - numpy array - array of length values for optical flow lines
        - lines - list - list of actual optical flow lines (where each line represents a trajectory of 
        a particular point in the image)
    '''

    angles = []
    translation = []

    h, w = img.shape[:2]    # 圖片高跟寬
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    
    for (x1, y1), (x2, y2) in lines:
        angle = math.atan2(- int(y2) + int(y1), int(x2) - int(x1)) * 180.0 / np.pi      # math.atan2輸出為弧度，將其轉成角度
        length = math.hypot(int(x2) - int(x1), - int(y2) + int(y1))
        translation.append(length)
        angles.append(angle)
    
    return np.array(angles), np.array(translation), lines

# function for drawing optical flow trajectories 
def draw_flow(img, lines):
    
    '''
    input:
        - img - numpy array - image to draw on
        - lines - list - list of lines to draw
        - BGR image with visualised optical flow
    '''

    width_delay_ratio = 6
    height_delay_ratio = 5
    
    h, w = img.shape[:2]
        
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    
    for (x1, y1), (x2, y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

    return vis

# function that analyses optical flow information
def estimate_motion(angles, translation):
    
    '''
    Input:
        - angles - numpy array - array of angles of optical flow lines to the x-axis
        - translation - numpy array - array of length values for optical flow lines
    Output:
        - ang_mode - float - mode of angles of trajectories. can be used to determine the direction of movement
        - transl_mode - float - mode of translation values 
        - ratio - float - shows how different values of translation are across a pair of frames. allows to 
        conclude about the type of movement
        - steady - bool - show if there is almost no movement on the video at the moment
    '''
    
    # Get indices of nonzero opical flow values. We'll use just them
    nonzero = np.where(translation > 0)
    
    # Whether non-zero value is close to zero or not. Should be set as a thershold
    steady = np.mean(translation) < 0.5
    
    translation = translation[nonzero]
    transl_mode = mode(translation)[0]      # 找出非0的translation中，出現最多次的數
    
    angles = angles[nonzero]
    ang_mode = mode(angles)[0]      # 找出非0的angles中，出現最多次的數
    ang_avg = np.average(angles)
    
    # cutt off twenty percent of the sorted list from both sides to get rid off outliers
    ten_percent = len(translation) // 10
    translations = sorted(translation)
    translations = translations[ten_percent: len(translations) - ten_percent]

    # cluster optical flow values and find out how different these cluster are
    # big difference (i.e. big ratio value) corresponds to panning, otherwise - trucking
    inliers = [tuple([inlier]) for inlier in translations]
    if len(inliers) < 3:
        return ang_mode, transl_mode, 0, steady
    k_means = KMeans(n_clusters=3, random_state=0).fit(inliers)
    centers = sorted(k_means.cluster_centers_)
    ratio = centers[0] / centers[-1]
    
    return ang_mode, ang_avg, transl_mode, ratio, steady

# specify directory and file name 
dir_path = "data"
filename = "419977240_6628933110545141_3090486628007442017_n.mp4"

# initialise stream from video
cap = cv.VideoCapture(os.path.join(dir_path, filename))
ret, prvs = cap.read()

# initialise video writer
frameRate = int(cap.get(cv.CAP_PROP_FPS))
codec = cv.VideoWriter_fourcc(*'mp4v')
save_name = "motion_" + filename[:-4] + ".mp4"
outputStream = cv.VideoWriter(save_name, codec, frameRate, (int(cap.get(3)),int(cap.get(4))))

# set parameters for text drawn on the frames
font = cv.FONT_HERSHEY_COMPLEX
fontScale = 1
fontColor = (68, 148, 213)
lineType  = 1

# initialise text variables to draw on frames
angle = 'None'
avg_angle = 'None'
translation = 'None'
motion = 'None'
motion_type = 'None'
direction = 'None'
# set counter value
count = 1
prev_time = time.time()

# main loop
while True:
    # read a new frame
    ret, nxt = cap.read()
    curr_time = time.time()
    time_diff = str(curr_time - prev_time)
    prev_time = curr_time

    if not ret:
        break
        
    # if the image is colored
    if len(prvs.shape) == 3:
        prvs_gray = cv.cvtColor(prvs.copy(), cv.COLOR_BGR2GRAY)
        next_gray = cv.cvtColor(nxt.copy(), cv.COLOR_BGR2GRAY)
    else:
        prvs_gray = prvs.copy()
        next_gray = nxt.copy()
        
    if count == 3:
        
        # calculate optical flow
        flow = cv.calcOpticalFlowFarneback(prvs_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # calculate trajectories and analyse them
        angles, transl, lines = calc_angl_n_transl(prvs_gray, flow)
        ang_mode, ang_avg, transl_mode, ratio, steady = estimate_motion(angles, transl)

        # draw trajectories on the frame
        next_gray = draw_flow(next_gray.copy(), lines)
#         next_gray = cv.cvtColor(next_gray.copy(), cv.COLOR_GRAY2BGR)
        
        if ang_mode >= -135 and ang_mode < -45:
            direction = 'Upward'
        elif ang_mode >= -45 and ang_mode < 45:
            direction = 'Left'
        elif ang_mode >= 135 or ang_mode < -135:
            direction = 'Right'
        else:
            direction = 'Downward'

        angle = str(round(ang_mode, 2))
        avg_angle = str(round(ang_avg,2))
        translation = str(round(transl_mode, 2))
        motion = 'No motion' if steady else round(ratio[0], 2)
        if isinstance(motion, float):
            motion_type = 'Panning' if motion > 0.6 else 'Trucking'
            
        count = 0
    
    # put values on the frame
    cv.putText(next_gray, direction, (50,50), font, fontScale, fontColor, lineType)
    cv.putText(next_gray, angle, (50,100), font, fontScale, fontColor, lineType)
    cv.putText(next_gray, avg_angle, (50,150), font, fontScale, fontColor, lineType)
    #cv.putText(next_gray, translation, (50,200), font, fontScale, fontColor, lineType)
    cv.putText(next_gray, motion_type, (50,200), font, fontScale, fontColor, lineType)
    #cv.putText(next_gray, time_diff, (50,250), font, fontScale, fontColor, lineType)
#     cv.putText(next_gray, str(motion), (50,90), font, fontScale, fontColor, lineType)
#     cv.putText(next_gray, motion_type, (50,150), font, fontScale, fontColor, lineType)
    
    # write the frame to the new video
    outputStream.write(next_gray)
    
    # update the previous frame
    prvs = nxt.copy()
    count += 1

outputStream.release()