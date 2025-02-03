import cv2
import imutils
import torch
from ultralytics import YOLO

yolo = YOLO("yolo11n-pose-better.pt")  # load a pretrained model (recommended for training)

'''
Here is the mapping of each index to its respective body joint:

0: Nose 1: Left Eye 2: Right Eye 3: Left Ear 4: Right Ear 5: Left Shoulder 6: Right Shoulder 7: Left Elbow 
8: Right Elbow 9: Left Wrist 10: Right Wrist 11: Left Hip 12: Right Hip 13: Left Knee 14: Right Knee 15: Left Ankle 16: Right Ankle

'''

# Load the video capture
videoCap = cv2.VideoCapture(0)

# Function to get class colors
def getColours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
    (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)


while True:
    ret, frame = videoCap.read()
    if not ret:
        continue
    results = yolo.track(frame, show=True, save=True)
    result = results[0]
    frame = results[0].plot()
    keypoints = result.keypoints.xy
    coordinatesList = torch.Tensor.tolist(keypoints)[0] #converts Tensor object to be iterable list
    coordinatesTuple = [tuple(i) for i in coordinatesList] #converts every coordinate from list to tuple
    coordinatesTuple = [(int(i[0]), int(i[1]))for i in coordinatesTuple] #returns every coordinate to be integer in order to prevent issues
    '''
    Here is the mapping of each index to its respective body joint:

    0: Nose 1: Left Eye 2: Right Eye 3: Left Ear 4: Right Ear 5: Left Shoulder 6: Right Shoulder 7: Left Elbow 
    8: Right Elbow 9: Left Wrist 10: Right Wrist 11: Left Hip 12: Right Hip 13: Left Knee 14: Right Knee 15: Left Ankle 
    16: Right Ankle
    '''
    

    

    cv2.rectangle(frame, (coordinatesTuple[1]), (coordinatesTuple[2]), (0, 0, 255), 2 )



    # show the image
    cv2.imshow('frame', frame)

    # break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the video capture and destroy all windows
videoCap.release()
cv2.destroyAllWindows()