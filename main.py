import os
import random

import numpy as np
import cv2
from ultralytics import YOLO
from tracker import Tracker
from scipy.spatial import ConvexHull

filename = "recording_one.mp4"

video_path = os.path.join('.', 'videos/Input/HornsFlare/', filename)
video_out_path = os.path.join('.', 'videos/Output/HornsFlare/', filename)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]))
defender_ids = [None] * 5
hasAddedDefenders = False

model = YOLO("yolov8n.pt") 
tracker = Tracker()

#10 players + 3 refs
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(13)]

#Areas output file
f = open("Areas/out.txt", 'w')

#Tracks output file
g = open("tracks/out.json", "w")
json_out = {"data":{}}
frame_num = 0
while ret:
    frame_num += 1
    results = model(frame)

    for person in results:
        detections = []
        for r in person.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_type = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_type = int(class_type)
            detections.append([x1, y1, x2, y2, confidence])

        tracker.update(frame, detections)

        vertices = []
        json_out["data"][frame_num] = {}
        for track in tracker.tracks:

            bbox = track.bbox
            track_id = track.track_id
            x1, y1, x2, y2 = bbox

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]))
            cv2.putText(frame, str(track_id),(int(x1), int(y1)),0, 5e-3 * 200, (colors[track_id % len(colors)]),2)

            #For tracked players, calculate center and associate it to vertex
            if hasAddedDefenders and track_id in defender_ids:
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                vertices.append([center_x, center_y])


                #Calculate velocity
                foundLastVal = False
                ind = frame_num - 1
                while not foundLastVal and ind > 0:
                    if track_id in json_out["data"][ind].keys():
                        velocity_x = (center_x - float(json_out["data"][ind][track_id][1:-1].split(",")[0])) / float(frame_num - ind)
                        velocity_y = center_y - float(json_out["data"][ind][track_id][1:-1].split(",")[1]) / float(frame_num - ind)
                        #Write location and velocity
                        json_out["data"][frame_num][track_id] = str([center_x, center_y, velocity_x, velocity_y])
                        foundLastVal = True
                    else:
                        ind -= 1
                if not foundLastVal:
                    json_out["data"][frame_num][track_id] = str([center_x, center_y, 0, 0])


    print("Writing tracks to file...")
    g.truncate(0)
    g.write(str(json_out))

    cv2.imshow('frame', frame)
    cv2.waitKey(25)
    cap_out.write(frame)
    ret, frame = cap.read()

    if hasAddedDefenders and len(vertices) >= 3:
        area = ConvexHull(vertices).area
        f.write(str(area) + "\n")

        for i in vertices:
            i[0], i[1] = int(i[0]), int(i[1])

        cnt = np.array(vertices)
        hull = cv2.convexHull(np.array(vertices, dtype='float32'))
        hull = np.array(hull).astype(np.int32)
        cv2.drawContours(frame, [hull], 0, (0, 255, 255), 3)
        cv2.putText(frame, "Area: " + str(area), vertices[-1], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    if not hasAddedDefenders:
        ids = input("Enter defender IDs separated by commas (e.g. 10,5,2,1,3 or space to skip): ")
        if ids == " ":
            continue
        else:
            defender_ids = [int(x) for x in ids.split(",")]
            hasAddedDefenders = True

cap.release()
cap_out.release()
cv2.destroyAllWindows()
f.close()
g.close()
