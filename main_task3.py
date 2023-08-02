# import libraries
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *
from video_writer import create_video

# Define yolo model, x is extra large model size
model=YOLO('yolov8x.pt')

# Define the regions of interest (ROI) coordinates
region1 = [(246, 424), (885, 357), (181, 5), (34, 4), (46,118)]  # Region 1 coordinates
region2 = [(1013, 375), (1369, 550), (1915,461), (1915, 173), (1075,198)]  # Region 2 coordinates
region3 = [(1359, 559), (442, 699), (588, 880), (1917, 873)]  # Region 3 coordinates

# Capture video from path
video_path = ('15.mp4')
cap=cv2.VideoCapture(video_path)
# to write the output video
output = create_video(cap, "15_output.mp4")

# Read class names from coco dataset file
file = open("config\coco.names", "r")
data = file.read()
# Define a list that contains all class names
class_list = data.split("\n") 

# Define a dictionary for vehicles leaving from region 1 to region 2 and 3
vehicle_1_2 = {}
from_1_2 = set()
vehicle_1_3 = {}
from_1_3 = set()

# Define a dictionary for vehicles leaving from region 2 to region 1 and 3
vehicle_2_1 = {}
from_2_1 = set()
vehicle_2_3 = {}
from_2_3 = set()

# Define a dictionary for vehicles leaving from region 3 to region 1 and 2
vehicle_3_1 = {}
from_3_1 = set()
vehicle_3_2 = {}
from_3_2 = set()

# Define object tracker
tracker = Tracker()

# Start tracking
while True:    
    ret,frame = cap.read()
    if not ret:
        break
    # Detect objects in the frame
    results=model.predict(frame)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    list=[]

    # Define the information of the bounding box related to each object
    for index,row in px.iterrows():
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]

        # list.append([x1, y1, x2, y2])
        # Add the vehicles objects coordination's to the list
        if 'bicycle' in c or 'car' in c or 'motorcycle' in c or 'bus' in c or 'train' in c or 'truck' in c or 'boat' in c:
            list.append([x1, y1, x2, y2])
    # Update the tracker
    bbox_id = tracker.update(list)
    # Define some conditions for each box
    for bbox in bbox_id:
        xmin, ymin, xmax, ymax, id = bbox
        center_x = int((xmin + xmax) / 2)
        center_y = int((ymin + ymax) / 2)
        # Define the points as tuples
        point1 = (xmin, ymin)
        point2 = (xmin, ymax)
        point3 = (xmax, ymax)
        point4 = (xmax, ymin)
        cen_point = (center_x, center_y)

        # Condition for vehicles moving from 1 to 2
        results = cv2.pointPolygonTest(np.array(region1, np.int32), (cen_point), False)
        if results >= 0:
            vehicle_1_2[id] = (xmax, ymax)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 204, 255), 2)
            cv2.circle(frame, (center_x, center_y), 4, (255,0,255), -1)
            cv2.putText(frame, str(id), (xmin + 65, ymin - 10), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255,0,255), 2)
        if id in vehicle_1_2:
            results1 = cv2.pointPolygonTest(np.array(region2, np.int32), (cen_point), False)
            if results1 >= 0:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 204, 255), 2)
                cv2.circle(frame, (center_x, center_y), 4, (255,0,255), -1)
                cv2.putText(frame, str(id), (xmin + 65, ymin - 10), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255,0,255), 2)
                from_1_2.add(id)
#-----------------------------------------------------------------------------------------------------------------------
        # Condition for vehicles moving from 1 to 3
        results2 = cv2.pointPolygonTest(np.array(region1, np.int32), (cen_point), False)
        if results2 >= 0:
            vehicle_1_3[id] = (xmax, ymax)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 204, 255), 2)
            cv2.circle(frame, (center_x, center_y), 4, (255,0,255), -1)
            cv2.putText(frame, str(id), (xmin + 65, ymin - 10), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255,0,255), 2)
        if id in vehicle_1_3:
            results3 = cv2.pointPolygonTest(np.array(region3, np.int32), (cen_point), False)
            if results3 >= 0:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 204, 255), 2)
                cv2.circle(frame, (center_x, center_y), 4, (255,0,255), -1)
                cv2.putText(frame, str(id), (xmin + 65, ymin - 10), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255,0,255), 2)
                from_1_3.add(id)
# -----------------------------------------------------------------------------------------------------------------------
        # Condition for vehicles moving from 2 to 1
        results4 = cv2.pointPolygonTest(np.array(region2, np.int32), (cen_point), False)
        if results4 >= 0:
            vehicle_2_1[id] = (xmax, ymax)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 204, 255), 2)
            cv2.circle(frame, (center_x, center_y), 4, (255,0,255), -1)
            cv2.putText(frame, str(id), (xmin + 65, ymin - 10), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255,0,255), 2)
        if id in vehicle_2_1:
            results5 = cv2.pointPolygonTest(np.array(region1, np.int32), (cen_point), False)
            if results5 >= 0:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 204, 255), 2)
                cv2.circle(frame, (center_x, center_y), 4, (255,0,255), -1)
                cv2.putText(frame, str(id), (xmin + 65, ymin - 10), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255,0,255), 2)
                from_2_1.add(id)
# -----------------------------------------------------------------------------------------------------------------------
        # Condition for vehicles moving from 2 to 3
        results6 = cv2.pointPolygonTest(np.array(region2, np.int32), (cen_point), False)
        if results6 >= 0:
            vehicle_2_3[id] = (xmax, ymax)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 204, 255), 2)
            cv2.circle(frame, (center_x, center_y), 4, (255,0,255), -1)
            cv2.putText(frame, str(id), (xmin + 65, ymin - 10), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255,0,255), 2)
        if id in vehicle_2_3:
            results7 = cv2.pointPolygonTest(np.array(region3, np.int32), (cen_point), False)
            if results7 >= 0:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 204, 255), 2)
                cv2.circle(frame, (center_x, center_y), 4, (255,0,255), -1)
                cv2.putText(frame, str(id), (xmin + 65, ymin - 10), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255,0,255), 2)
                from_2_3.add(id)
# -----------------------------------------------------------------------------------------------------------------------
        # Condition for vehicles moving from 3 to 1
        results8 = cv2.pointPolygonTest(np.array(region3, np.int32), (cen_point), False)
        if results8 >= 0:
            vehicle_3_1[id] = (xmax, ymax)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 204, 255), 2)
            cv2.circle(frame, (center_x, center_y), 4, (255,0,255), -1)
            cv2.putText(frame, str(id), (xmin + 65, ymin - 10), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255,0,255), 2)
        if id in vehicle_3_1:
            results9 = cv2.pointPolygonTest(np.array(region1, np.int32), (cen_point), False)
            if results9 >= 0:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 204, 255), 2)
                cv2.circle(frame, (center_x, center_y), 4, (255,0,255), -1)
                cv2.putText(frame, str(id), (xmin + 65, ymin - 10), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255,0,255), 2)
                from_3_1.add(id)
# -----------------------------------------------------------------------------------------------------------------------
        # Condition for vehicles moving from 3 to 2
        results10 = cv2.pointPolygonTest(np.array(region3, np.int32), (cen_point), False)
        if results10 >= 0:
            vehicle_3_2[id] = (xmax, ymax)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 204, 255), 2)
            cv2.circle(frame, (center_x, center_y), 4, (255,0,255), -1)
            cv2.putText(frame, str(id), (xmin + 65, ymin - 10), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255,0,255), 2)
        if id in vehicle_3_2:
            results11 = cv2.pointPolygonTest(np.array(region2, np.int32), (cen_point), False)
            if results11 >= 0:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 204, 255), 2)
                cv2.circle(frame, (center_x, center_y), 4, (255,0,255), -1)
                cv2.putText(frame, str(id), (xmin + 65, ymin - 10), cv2.FONT_HERSHEY_COMPLEX, (0.5), (255,0,255), 2)
                from_3_2.add(id)
# -----------------------------------------------------------------------------------------------------------------------
    # Write the output frame
    output.write(frame)

    # In case we want to show the borders of the ROIs
    # cv2.polylines(frame,[np.array(region1,np.int32)],True,(0,0,0),2)
    # cv2.polylines(frame,[np.array(region2,np.int32)],True,(0,0,0),2)
    # cv2.polylines(frame,[np.array(region3,np.int32)],True,(0,0,0),2)

    # Define the final results based on the length of the sets
    count_1_2 = len(from_1_2)
    count_1_3 = len(from_1_3)
    count_2_1 = len(from_2_1)
    count_2_3 = len(from_2_3)
    count_3_1 = len(from_3_1)
    count_3_2 = len(from_3_2)

    # To write the counts on the frame itself
    cv2.putText(frame,'1-2 '+str(count_1_2),(10,30),cv2.FONT_HERSHEY_COMPLEX,(1),(255, 204, 255),2)
    cv2.putText(frame,'1-3 '+str(count_1_3),(10,60),cv2.FONT_HERSHEY_COMPLEX,(1),(201,255,135),2)
    cv2.putText(frame,'2-1 '+str(count_2_1),(10,90),cv2.FONT_HERSHEY_COMPLEX,(1),(204,255,255),2)
    cv2.putText(frame,'2-3 '+str(count_2_3),(10,120),cv2.FONT_HERSHEY_COMPLEX,(1),(0,0,153),2)
    cv2.putText(frame,'3-1 '+str(count_3_1),(10,150),cv2.FONT_HERSHEY_COMPLEX,(1),(102,0,204),2)
    cv2.putText(frame,'3-2 '+str(count_3_2),(10,180),cv2.FONT_HERSHEY_COMPLEX,(1),(50,150,255),2)

    # To show the tracking and counting operation
    cv2.imshow("Tracking & Counting", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
# To write the results in new text file
base_name = video_path.split(".")[0]
text_filename = base_name + "_predicted.txt"
with open(text_filename, "w") as file:
    file.write("1-2 " + str(count_1_2) + "\n")
    file.write("1-3 " + str(count_1_3) + "\n")
    file.write("2-1 " + str(count_2_1) + "\n")
    file.write("2-3 " + str(count_2_3) + "\n")
    file.write("3-1 " + str(count_3_1) + "\n")
    file.write("3-2 " + str(count_3_2) + "\n")
    file.close()

cap.release()
output.release()
cv2.destroyAllWindows()