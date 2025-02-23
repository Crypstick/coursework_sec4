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
prev_filter_params_face = {}  # Dictionary to store parameters for each person
prev_filter_params_body = {}  

# load ledaddy face oh yeahhhhhh
class Image:
    def __init__(self, file, height, width):
        self.file = file
        self.height = height
        self.width = width

CLEAR = 0
SHIRT = 1
FACE = 2
PIC_URL = ["empty.png", "lebron.png", "hat.png", "sstshirt.png", "bluesw.png"]
PIC_TYPE = [CLEAR, FACE, FACE, SHIRT, SHIRT]
PIC_ICON_SIZE = (100, 100)
MARGIN_FACE = 100
MARGIN_BODY = 300
pictures = [] #reference for the picture 1. file 2. height 3. width
for i in range(len(PIC_URL)):
    tmp = cv2.imread(PIC_URL[i], -1)
    if tmp is None or tmp.shape[0] == 0 or tmp.shape[1] == 0:
        raise FileNotFoundError("aint no",str)
    else:
        h, w = PIC_ICON_SIZE
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


#for the snapping; js 1. position of the nose 2. assigned thing
class Person:
    def __init__(self, face_image_it, face_x, face_y, body_image_it, body_x, body_y):
        self.face_image_it = face_image_it
        self.face_x = face_x
        self.face_y = face_y
        self.body_image_it = body_image_it
        self.body_x = body_x
        self.body_y = body_y


people = {}
amt_people = 0

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
            #print(SELECTION_DEFAULT[dragging_obj].x, SELECTION_DEFAULT[dragging_obj].y)
            selection_current[dragging_obj].x = SELECTION_DEFAULT[dragging_obj].x
            selection_current[dragging_obj].y = SELECTION_DEFAULT[dragging_obj].y
            
            # assign the thing to someone
            if PIC_TYPE[dragging_obj] == SHIRT or PIC_TYPE[dragging_obj] == CLEAR:
                for i in people:
                    if abs(x-people[i].body_x < MARGIN_BODY) and abs(y-people[i].body_y < MARGIN_BODY):
                        people[i].body_image_it = dragging_obj
                        print("body" , i)
                        break
            
            if PIC_TYPE[dragging_obj] == FACE or PIC_TYPE[dragging_obj] == CLEAR:
                for i in people:
                    if abs(x-people[i].face_x < MARGIN_FACE) and abs(y-people[i].face_y < MARGIN_FACE):
                        people[i].face_image_it = dragging_obj
                        print("face" , i)
                        break
            dragging_obj = None
        #decide if the mouse click is close enuough to the picture to be clicked
        else:
            for i in range(len(pictures)):
                #print(i,x, SELECTION_DEFAULT[i].x, y, SELECTION_DEFAULT[i].y)
                if x-SELECTION_DEFAULT[i].x > 0 and x-SELECTION_DEFAULT[i].x < pictures[i].width:
                    if y-SELECTION_DEFAULT[i].y > 0 and y-SELECTION_DEFAULT[i].y < pictures[i].height:
                        drag_state = True
                        dragging_obj = i
                        break
    #when dragin update position
    elif event == cv2.EVENT_MOUSEMOVE and drag_state:
        selection_current[dragging_obj].x = x-(pictures[dragging_obj].width//2)
        selection_current[dragging_obj].y = y-(pictures[dragging_obj].height//2)
    #print(event, x, y)
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

def overlay_shirt(kp):
    if len(kp) < 13:
        return
    # Get shoulder points with better precision
    left_shoulder = tuple(map(int, kp[5]))
    right_shoulder = tuple(map(int, kp[6]))
    # Get hip points for better proportions
    left_hip = tuple(map(int, kp[11]))
    right_hip = tuple(map(int, kp[12]))

    # Calculate shirt dimensions with improved proportions and validation
    shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
    torso_height = abs((left_hip[1] + right_hip[1])/2 - (left_shoulder[1] + right_shoulder[1])/2)

    # Ensure minimum dimensions to prevent resize errors
    shirt_width = max(50, int(shoulder_width * 2.5))  # Minimum width of 50 pixels
    shirt_height = max(50, int(torso_height * 2.0))  # Minimum height of 50 pixels

    # Enhanced shirt positioning using midpoints
    shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) // 2
    shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) // 2
    shirt_x = int(shoulder_center_x - shirt_width // 2)
    shirt_y = int(shoulder_center_y - shirt_height * 0.3)  # Higher placement

    # Skip if dimensions are invalid
    if shirt_width <= 0 or shirt_height <= 0:
        return

    # Improved angle calculation using both shoulders
    shoulder_angle = np.degrees(np.arctan2(right_shoulder[1] - left_shoulder[1],
                                        right_shoulder[0] - left_shoulder[0]))
    shirt_angle = 0  # Keep shirt perfectly upright

    # Apply the overlay with adjusted parameters
    return (shirt_x, shirt_y, shirt_width, shirt_height, shirt_angle)


#only process the yolo every 4 frames
last_seen = []
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
        last_seen = []
        # run yolo inference on the frame with higher confidence
        #frame[:][:481] cause the bottom 200px is js my rectangle
        results = model(frame[:][:481], conf=0.3)  # lower confidence threshold for better detection
        # Process body detections
        if hasattr(results[0], "keypoints") and results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy.cpu().numpy()

            # THIS PART WAS SO FUCKING DIFFICULT WTF BRO
            #person_idx = -1
            for person_id, kp in enumerate(keypoints):  # loop through all detected people with unique index
                if len(kp) < 13:
                    continue
                else:
                    #update each persons stats for the drag and drop
                    last_seen.append(person_id)
                    if (not person_id in people):
                        people[person_id] = Person(0, 0, 0, 0, 0, 0)
                        amt_people += 1
                    #face
                    keypoints_int = kp[:13].astype(np.int32)
                    nose_x, nose_y = keypoints_int[0]
                    people[person_id].face_x = nose_x
                    people[person_id].face_y = nose_y

                    #torso
                    l_shoulder_x, l_shoulder_y = keypoints_int[5]
                    people[person_id].body_x = l_shoulder_x
                    people[person_id].body_y = l_shoulder_y


                    #now add overlays
                    if (people[person_id].body_image_it != 0):
                        res = overlay_shirt(kp)
                        if res == None:
                            body_x = body_y = body_width = body_height = body_angle = 0
                        else:
                            body_x, body_y, body_width, body_height, body_angle = res
                            # boundary checks
                            body_x = max(0, min(body_x, frame.shape[1] - body_width))
                            body_y = max(0, min(body_y, frame.shape[0] - body_height))
                            #hawk tuah overlay that thing
                            frame = overlay_image(frame, pictures[people[person_id].body_image_it].file, body_x, body_y, (body_width, body_height), body_angle)
                    else:
                        body_x = body_y = body_width = body_height = body_angle = 0

                    if (people[person_id].face_image_it != 0):
                        res = overlay_LEBRON(kp)
                        if res == None:
                            face_x = face_y = face_width = face_height = face_angle = 0
                        else: 
                            face_x, face_y, face_width, face_height, face_angle = res
                            # boundary checks
                            face_x = max(0, min(face_x, frame.shape[1] - face_width))
                            face_y = max(0, min(face_y, frame.shape[0] - face_height))
                            #hawk tuah overlay that thing
                            frame = overlay_image(frame, pictures[people[person_id].face_image_it].file, face_x, face_y, (face_width, face_height), face_angle)
                    else:
                        face_x = face_y = face_width = face_height = face_angle = 0

                    

                    # store parameters for each person
                    prev_filter_params_face[person_id] = (face_x, face_y, face_width, face_height, face_angle)
                    prev_filter_params_body[person_id] = (body_x, body_y, body_width, body_height, body_angle)
                    
                

                    
                    # Draw all keypoints and connections for the full body
                    # Define the keypoint connections for the body
                    skeleton = [
                        [5, 7], [7, 9],     # Right arm
                        [6, 8], [8, 10],    # Left arm
                        [5, 6],             # Shoulders
                        [5, 11], [6, 12],   # Torso
                        [11, 13], [13, 15], # Right leg
                        [12, 14], [14, 16], # Left leg
                        [11, 12],           # Hips
                        [0, 1], [0, 2],     # Nose to eyes
                        [1, 3], [2, 4],     # Eyes to ears
                    ]

                    # Draw all keypoints
                    for idx, (x, y) in enumerate(kp):
                        color = (0, 255, 0) if idx < 5 else (0, 255, 255)  # Green for face, Cyan for body
                        cv2.circle(frame, (int(x), int(y)), 5, color, -1)

                    # Draw skeleton connections
                    for start_idx, end_idx in skeleton:
                        if start_idx < len(kp) and end_idx < len(kp):
                            start_point = tuple(map(int, kp[start_idx]))
                            end_point = tuple(map(int, kp[end_idx]))
                            cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
                    
                    
                '''
                # smooth transitions for each person independently (face only)
                if person_idx in prev_filter_params:
                    prev_x, prev_y, prev_w, prev_h, prev_angle = prev_filter_params[person_idx]
                    filter_x = int(0.3 * prev_x + 0.5 * new_filter_x)  
                    filter_y = int(0.5 * prev_y + 0.5 * new_filter_y)
                    filter_width = int(0.5 * prev_w + 0.5 * filter_width)
                    filter_height = int(0.5 * prev_h + 0.5 * filter_height)
                else:
                    filter_x = new_filter_x
                    filter_y = new_filter_y
                '''

                

                
                    
    else:
        #overlay old ones when not processing
        for person_id in last_seen:
            if (people[person_id].body_image_it != 0 and person_id in prev_filter_params_body):
                filter_x, filter_y, filter_width, filter_height, angle = prev_filter_params_body[person_id]
                if (filter_width != 0):
                    frame = overlay_image(frame, pictures[people[person_id].body_image_it].file, filter_x, filter_y, (filter_width, filter_height), angle)

            if (people[person_id].face_image_it != 0  and person_id in prev_filter_params_face):
                filter_x, filter_y, filter_width, filter_height, angle = prev_filter_params_face[person_id]
                if (filter_width != 0):
                    frame = overlay_image(frame, pictures[people[person_id].face_image_it].file, filter_x, filter_y, (filter_width, filter_height), angle)
            
            
                
    #create the selection area
    frame = cv2.rectangle(frame, (0,FRAME_HEIGHT-SELECTION_AREA_HEIGHT), (FRAME_WIDTH-1,FRAME_HEIGHT-1), (128,128, 128),-1)
    frame = cv2.rectangle(frame, (0,FRAME_HEIGHT-SELECTION_AREA_HEIGHT), (FRAME_WIDTH-1,FRAME_HEIGHT-1), (255,0, 0),5)
    #overlay the selections
    for i in range(len(selection_current)):
        frame = overlay_image(frame, pictures[i].file, selection_current[i].x, selection_current[i].y, (pictures[i].height, pictures[i].width), 0)

    #
    if (drag_state):
        for person_id in last_seen:
            if PIC_TYPE[dragging_obj] == FACE or PIC_TYPE[dragging_obj] == CLEAR:
                frame = cv2.circle(frame, (people[person_id].face_x, people[person_id].face_y), MARGIN_FACE,  (0, 0, 255), 2)
            if PIC_TYPE[dragging_obj] == SHIRT or PIC_TYPE[dragging_obj] == CLEAR:
                frame = cv2.circle(frame, (people[person_id].body_x, people[person_id].body_y), MARGIN_BODY,  (0, 0, 255), 2)

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

