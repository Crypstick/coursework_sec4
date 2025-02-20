import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time

# load yolo model
model = YOLO("yolo11n-pose-better.pt")  


# hijack dat mufuckin webcam bitchhhhh
cap = cv2.VideoCapture(0)
FRAME_TITLE = "LeBron Filter + Full Tracking View (Multi-Person + Head Tilt Fixed)"
FRAME_WIDTH = 640 # making it small for speed
FRAME_HEIGHT = 480
SELECTION_AREA_HEIGHT = 125
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)  # gotta go fast

cv2.namedWindow(FRAME_TITLE)

# some optimisation shit that chatgpt suggested
prev_frame_time = 0
new_frame_time = 0
prev_filter_params = {}  # Dictionary to store parameters for each person

# load ledaddy face oh yeahhhhhh
class Image:
    def __init__(self, file, height, width):
        self.file = file
        self.height = height
        self.width = width

PIC_URL = ["lebron.png", "hat.png"]
#define costom size, use None if you want to use original size
PIC_SIZE_OVERRIDE = [(100,100),(100,100)] 
pictures = [] #reference for the picture 1. file 2. height 3. width
for i in range(len(PIC_URL)):
    tmp = cv2.imread(PIC_URL[i], -1)
    if tmp is None or tmp.shape[0] == 0 or tmp.shape[1] == 0:
        raise FileNotFoundError("aint no",str)
    else:
        if PIC_SIZE_OVERRIDE[i] != None:
            h, w = PIC_SIZE_OVERRIDE[i]
        else:
            h, w, c = tmp.shape
        pictures.append(Image(tmp,h,w))

#Location of the selection pictures
# auto assign them to spread out in the selection frame
class Coord:
    def __init__(self, x, y):
        self.x = x
        self.y = y
interval_x = FRAME_WIDTH // (len(pictures)+1)
interval_y = FRAME_HEIGHT-SELECTION_AREA_HEIGHT//2
SELECTION_DEFAULT = [] #where pictures should return to
selection_current = [] #current location of where they are, even when they are moving
for i in range(len(pictures)):
    SELECTION_DEFAULT.append(Coord(interval_x*(i+1)-pictures[i].width//2, interval_y-pictures[i].height//2))
    selection_current.append(Coord(SELECTION_DEFAULT[i].x, SELECTION_DEFAULT[i].y))
print(selection_current)

#for the snapping; js 1. position of the nose 2. assigned thing
class Person:
    def __init__(self, image_it, x, y):
        self.image_it = image_it
        self.x = x
        self.y = y
        self.new_filter_x = y
        self.y = y
        self.y = y
        self.y = y


people = [Person(-1, 0, 0)]
amt_people = 1

# making filter tilt when we tilt our head
def rotate_image(image, angle):
  
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)  # 🔄 No more upside-down LeBron!
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return rotated

# hawk tuah that thing
def overlay_image(background, overlay, x, y, overlay_size, angle):
    """
    The magical face-swapping function! ✨
    """
    overlay = cv2.resize(overlay, overlay_size)
    overlay = rotate_image(overlay, angle)  

    h, w, c = overlay.shape

    # beep beep out of bounds bitch
    x = max(0, min(int(x), background.shape[1] - 1))
    y = max(0, min(int(y), background.shape[0] - 1))

    if y + h > background.shape[0]:  
        h = background.shape[0] - y
        overlay = overlay[:h, :, :]  

    if x + w > background.shape[1]:  
        w = background.shape[1] - x
        overlay = overlay[:, :w, :]  

    # some transparency shi that chatgpt told me to do
    if c == 3:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)

    # pixel blending
    alpha_overlay = overlay[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_overlay

    for c in range(0, 3):  
        background[y:y+h, x:x+w, c] = (
            alpha_overlay[:h, :w] * overlay[:h, :w, c] +
            alpha_background[:h, :w] * background[y:y+h, x:x+w, c]
        )

    return background

#handle most of the selection logic
drag_state = False
dragging_obj = None
def change_pic_coord(event,x,y,flags,param):
    global drag_state, dragging_obj, pictures, selection_current, SELECTION_DEFAULT, people, amt_people
    if event == cv2.EVENT_LBUTTONDOWN:
         #put down the picture
        if drag_state:
            drag_state = False
            #reset the selector picture to go back
            print(SELECTION_DEFAULT[dragging_obj].x, SELECTION_DEFAULT[dragging_obj].y)
            selection_current[dragging_obj].x = SELECTION_DEFAULT[dragging_obj].x
            selection_current[dragging_obj].y = SELECTION_DEFAULT[dragging_obj].y
            
            # assign the thing to someone
            MARGIN = 150
            for i in range(amt_people):
                if abs(x-people[i].x < MARGIN) and abs(y-people[i].y < MARGIN):
                    people[i].image_it = dragging_obj
                    print("BANGGG" , i)
                    break
            dragging_obj = None
        #decide if the mouse click is close enuough to the picture to be clicked
        else:
            for i in range(len(pictures)):
                print(i,x, SELECTION_DEFAULT[i].x, y, SELECTION_DEFAULT[i].y)
                if x-SELECTION_DEFAULT[i].x > 0 and x-SELECTION_DEFAULT[i].x < pictures[i].width:
                    if y-SELECTION_DEFAULT[i].y > 0 and y-SELECTION_DEFAULT[i].y < pictures[i].height:
                        drag_state = True
                        dragging_obj = i
                        break
    #when dragin update position
    elif event == cv2.EVENT_MOUSEMOVE and drag_state:
        selection_current[dragging_obj].x = x-(pictures[dragging_obj].width//2)
        selection_current[dragging_obj].y = y-(pictures[dragging_obj].height//2)
    print(event, x, y)

cv2.setMouseCallback(FRAME_TITLE,change_pic_coord)

def overlay_LEBRON(kp):
    try:
        # convert keypoints to integers more efficiently
        keypoints_int = kp[:5].astype(np.int32)
        nose_x, nose_y = keypoints_int[0]
        left_eye_x, left_eye_y = keypoints_int[1]
        right_eye_x, right_eye_y = keypoints_int[2]
        left_ear_x, left_ear_y = keypoints_int[3]
        right_ear_x, right_ear_y = keypoints_int[4]  
        
        # estimate chin using eye-nose distance
        eye_mid_x = int((left_eye_x + right_eye_x) // 2)
        eye_mid_y = int((left_eye_y + right_eye_y) // 2)

        # distance from nose to eyes
        nose_to_eye_distance = abs(nose_y - eye_mid_y)

        # estimate chin position
        chin_x = eye_mid_x
        chin_y = int(nose_y + (nose_to_eye_distance * 2.5))  # Adjust multiplier for better accuracy

        # calculate head tilt angle
        angle = np.degrees(np.arctan2(left_eye_y - right_eye_y, left_eye_x - right_eye_x))

    except IndexError:
        return  

    # more lenient ear position check
    if left_ear_y > chin_y or right_ear_y > chin_y:
        return  

    # more lenient chin position check
    if chin_y < nose_y or chin_y - nose_y > 400:  # Increased threshold
        return  

    # more lenient ear spacing check
    ear_distance = abs(right_ear_x - left_ear_x)
    if ear_distance < 15 or ear_distance > 400:  # Adjusted thresholds
        return  

    # improved face width calculation
    face_width = ear_distance
    face_height = abs(chin_y - eye_mid_y) * 2.2  # making sure the crown fits just right

    # size matters
    filter_width = max(120, int(face_width * 1.3))  # make room for the beard
    filter_height = max(140, int(face_height * 1.6))  # tall head

    # perfect placement
    new_filter_x = nose_x - filter_width // 2
    new_filter_y = nose_y - int(filter_height * 0.6) 
    return (new_filter_x,new_filter_y, filter_width,filter_height, angle)

frame_count = 0
frame_skip = 3
while cap.isOpened():
    ret, frame = cap.read()
    
    # mirroed view cos my asymetrical face got me fucked up gng
    frame = cv2.flip(frame, 1)
    
    # calculate FPS
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 0
    prev_frame_time = new_frame_time
    
    # skip frames if processing is too slow
    if fps < 15 and fps > 0:
        continue
    if not ret:
        break
    frame_count += 1
    if frame_count == frame_skip:
        frame_count = 0
        # run yolo inference on the frame with higher confidence
        #frame[:][:481] cause the bottom 200px is js my rectangle
        results = model(frame[:][:481], conf=0.3)  # lower confidence threshold for better detection
        # Process body detections
        if hasattr(results[0], "keypoints") and results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy.cpu().numpy()

            # THIS PART WAS SO FUCKING DIFFICULT WTF BRO
            amt_people = 0
            person_idx = -1
            for person_id, kp in enumerate(keypoints):  # loop through all detected people with unique index
                res = overlay_LEBRON(kp)
                if res == None:
                    continue
                else:
                    person_idx +=1
                    keypoints_int = kp[:5].astype(np.int32)
                    nose_x, nose_y = keypoints_int[0]
                    amt_people += 1
                    if (len(people) < amt_people):
                        people.append(Person(-1, 0, 0))
                    people[person_idx].x = nose_x
                    people[person_idx].y = nose_y
                    new_filter_x, new_filter_y, filter_width, filter_height, angle = res

                # smooth transitions for each person independently
                if person_idx in prev_filter_params:
                    prev_x, prev_y, prev_w, prev_h, prev_angle = prev_filter_params[person_idx]
                    filter_x = int(0.3 * prev_x + 0.5 * new_filter_x)  
                    filter_y = int(0.5 * prev_y + 0.5 * new_filter_y)
                    filter_width = int(0.5 * prev_w + 0.5 * filter_width)
                    filter_height = int(0.5 * prev_h + 0.5 * filter_height)
                else:
                    filter_x = new_filter_x
                    filter_y = new_filter_y

                # store parameters for each person
                prev_filter_params[person_idx] = (filter_x, filter_y, filter_width, filter_height, angle)

                # boundary checks
                filter_x = max(0, min(filter_x, frame.shape[1] - filter_width))
                filter_y = max(0, min(filter_y, frame.shape[0] - filter_height))

                # LEBRONIFICATION
                #if the imagie it == -1, no image
                if (people[person_idx].image_it != -1):
                    frame = overlay_image(frame, pictures[people[person_idx].image_it].file, filter_x, filter_y, (filter_width, filter_height), angle)
    else:
        for person_idx in range(amt_people):
            if (people[person_idx].image_it != -1):
                filter_x, filter_y, filter_width, filter_height, angle = prev_filter_params[person_idx]
                frame = overlay_image(frame, pictures[people[person_idx].image_it].file, filter_x, filter_y, (filter_width, filter_height), angle)
    #create the selection area
    frame = cv2.rectangle(frame, (0,FRAME_HEIGHT-SELECTION_AREA_HEIGHT), (FRAME_WIDTH-1,FRAME_HEIGHT-1), (128,128, 128),-1)
    frame = cv2.rectangle(frame, (0,FRAME_HEIGHT-SELECTION_AREA_HEIGHT), (FRAME_WIDTH-1,FRAME_HEIGHT-1), (255,0, 0),5)
    for i in range(len(selection_current)):
        frame = overlay_image(frame, pictures[i].file, selection_current[i].x, selection_current[i].y, (pictures[i].height, pictures[i].width), 0)

    # showing FPS
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, 'hold q to quit', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow(FRAME_TITLE, frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# thats game
cap.release()
cv2.destroyAllWindows()
