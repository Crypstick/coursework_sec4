import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time

# load yolo model
model = YOLO("yolo11n-pose-better.pt")  

# hijack dat webcam 
cap = cv2.VideoCapture(0)
FRAME_TITLE = "LeBron Filter + Full Tracking View (Multi-Person + Head Tilt Fixed)"
FRAME_WIDTH = 640 # making it small for speed
FRAME_HEIGHT = 480
SELECTION_AREA_WIDTH = 100
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)  # gotta go fast
cv2.namedWindow(FRAME_TITLE)

#fps skipping global vars
prev_frame_time = 0
new_frame_time = 0

# dictionary of peoples current filters + location for the snapping
class Person:
    def __init__(self, face_image_it, face_x, face_y, body_image_it, body_x, body_y, hand_image_it, l_hand_x, l_hand_y, r_hand_x, r_hand_y):
        self.face_image_it = face_image_it
        self.face_x = face_x
        self.face_y = face_y
        self.body_image_it = body_image_it
        self.body_x = body_x
        self.body_y = body_y
        self.hand_image_it = hand_image_it
        self.l_hand_x = l_hand_x
        self.l_hand_y = l_hand_y
        self.r_hand_x = r_hand_x
        self.r_hand_y = r_hand_y
people = {}

# dictionary to store previous filter positions for each person + smoothening
prev_filter_params_face = {} 
prev_filter_params_body = {}
prev_filter_params_hands_l = {}
prev_filter_params_hands_r = {}  


# create picture list
class Image:
    def __init__(self, file, height, width):
        self.file = file
        self.height = height
        self.width = width
pictures = [] #list of Image --> 1. file 2. height 3. width
CLEAR = 0 #quick enums
SHIRT = 1
FACE = 2
HAND = 3
PIC_URL = ["empty.png", "heart2.png", "birthday_hat_transparent.png", "sstshirt.png", "bluesw.png", "redsw.png", "greensw.png", "blacksw.png", "yellowsw.png", "sst_jacket_torso.png", "transparent_gauntlet.png"]
PIC_TYPE = [CLEAR, FACE, FACE, SHIRT, SHIRT, SHIRT, SHIRT, SHIRT, SHIRT, SHIRT, HAND]
PIC_ICON_SIZE_DEFAULT = (70, 50)
PIC_ICON_SIZE = [None, (70, 40), None, None, None, None, None, None, None, None, (70, 70)]
for i in range(len(PIC_URL)): #load pictures into the list
    tmp = cv2.imread("pictures/"+PIC_URL[i], -1)
    if tmp is None or tmp.shape[0] == 0 or tmp.shape[1] == 0:
        raise FileNotFoundError("aint no",str)
    else:
        if (PIC_ICON_SIZE[i] == None):
            h, w = PIC_ICON_SIZE_DEFAULT
        else:
            h, w = PIC_ICON_SIZE[i]
        pictures.append(Image(tmp,h,w))

# filter management
#face_filters = {0: None, 1: lebron_face, 2: heart_face, 3: birthday_hat}  # added birthday hat filter
face_width_multipliers = {1: 1.2, 2: 1.4}  # width multipliers for each face filter, wider for hat
face_height_multipliers = {1: 0.4, 2: 0.6}  # height multipliers for each face filter, shorter for hat
#shirt_filters = {0: None, 1: tshirt, 2: blue_shirt, 3: red_shirt, 4: green_shirt, 5: black_shirt, 6: yellow_shirt, 7: jacket_torso}  # Use jacket_torso as base image
shirt_offsets = {3: 0.3, 4: 0.3, 5: 0.2, 6: 0.3, 7: 0.3, 8: 0.3, 9: 0.3}  # adjusted offset for jacket_torso
shirt_width_multipliers = {3: 3, 4: 2.5, 5: 2.0, 6: 2.5, 7: 2.5, 8: 2.5, 9: 2.5}  # adjusted width multiplier for jacket_torso
shirt_height_multipliers = {3: 2.5, 4: 2, 5: 1.6, 6: 2, 7: 2, 8: 2, 9: 2}  # adjusted height multiplier for jacket_torso
#hand_filters = {0: None, 1: gauntlet}  # add gauntlet filter for hands
hand_width_multipliers = {10: 2.5} 
hand_height_multipliers = {10: 2.5}  
current_face_filter = 0
current_body_filter = 0
current_hand_filter = 0

#Location of the selection pictures
# auto assign them to spread out in the selection frame
class Coord:
    def __init__(self, x, y):
        self.x = x
        self.y = y
interval_y = FRAME_HEIGHT // (len(pictures)+1)
interval_x = FRAME_WIDTH-SELECTION_AREA_WIDTH//2
SELECTION_DEFAULT = [] #where pictures should return to
selection_current = [] #current location of where they are, even when they are moving
for i in range(len(pictures)):
    SELECTION_DEFAULT.append(Coord(interval_x-pictures[i].width//2, interval_y*(i+1)-pictures[i].height//2))
    selection_current.append(Coord(SELECTION_DEFAULT[i].x, SELECTION_DEFAULT[i].y))
#margins for the selection
MARGIN_FACE = 70
MARGIN_BODY = 100
MARGIN_HAND = 50

#handle most of the selection logic
drag_state = False
dragging_obj = None
def change_pic_coord(event,x,y,flags,param):
    global drag_state, dragging_obj, pictures, selection_current, SELECTION_DEFAULT, people
    if event == cv2.EVENT_LBUTTONDOWN:
         #put down the picture
        if drag_state:
            drag_state = False
            #reset the selector picture to go back
            #print(SELECTION_DEFAULT[dragging_obj].x, SELECTION_DEFAULT[dragging_obj].y)
            selection_current[dragging_obj].x = SELECTION_DEFAULT[dragging_obj].x
            selection_current[dragging_obj].y = SELECTION_DEFAULT[dragging_obj].y
            
            clear_avl = True
            # assign the thing to someone
            for i in last_seen:
                if PIC_TYPE[dragging_obj] == SHIRT or (PIC_TYPE[dragging_obj] == CLEAR and clear_avl):
                    if abs(x-people[i].body_x) < MARGIN_BODY and abs(y-people[i].body_y) < MARGIN_BODY:
                        people[i].body_image_it = dragging_obj
                        print("body" , i)
                        print(abs(x-people[i].body_x) < MARGIN_BODY, abs(y-people[i].body_y) < MARGIN_BODY)
                        #clear_avl = False
                        break
            
                if PIC_TYPE[dragging_obj] == FACE or (PIC_TYPE[dragging_obj] == CLEAR and clear_avl):
                    if abs(x-people[i].face_x) < MARGIN_FACE and abs(y-people[i].face_y) < MARGIN_FACE:
                        people[i].face_image_it = dragging_obj
                        print("face" , i)
                        #clear_avl = False
                        break

                if PIC_TYPE[dragging_obj] == HAND or (PIC_TYPE[dragging_obj] == CLEAR and clear_avl):
                    if ( (abs(x-people[i].l_hand_x) < MARGIN_HAND and abs(y-people[i].l_hand_y) < MARGIN_HAND) or (abs(x-people[i].r_hand_x) < MARGIN_HAND and abs(y-people[i].r_hand_y) < MARGIN_HAND)):
                        people[i].hand_image_it = dragging_obj
                        print("hand" , i)
                        #clear_avl = False
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




# making filter tilt when we tilt our head
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)  # fixed upside down lebron
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

    # beep beep out of bounds
    x = max(0, min(int(x), background.shape[1] - 1))
    y = max(0, min(int(y), background.shape[0] - 1))

    if y + h > background.shape[0]:  
        h = background.shape[0] - y
        overlay = overlay[:h, :, :]  

    if x + w > background.shape[1]:  
        w = background.shape[1] - x
        overlay = overlay[:, :w, :]  


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



#show instructions
instructions = ["1.png", "2.png", "3.png", "4.png", "5.png"]
click = False
cv2.namedWindow("Instructions")
def clicked(event,x,y,flags,param):
    global click
    if event == cv2.EVENT_LBUTTONDOWN:
        click = True
cv2.setMouseCallback("Instructions", clicked)
for page in instructions:
    tmp = cv2.imread("instructions/"+page, -1)
    cv2.imshow("Instructions", tmp)
    while (not click):
        key =  cv2.waitKey(1)
        if key == ord("\r") or  key == ord("\n"):
            break
    click = False
cv2.destroyWindow("Instructions")

#we only process yolo every other frames so the mouse can actually catch inputs
frame_count = 0
frame_skip = 2
last_seen = [] #key of people we saw when processing

while cap.isOpened():
    ret, frame = cap.read()
    
    # mirroed view 
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

    #is it the 3rd frame yet??
    frame_count += 1
    if frame_count == frame_skip:
        frame_count = 0
        last_seen = []

        # run yolo inference on the frame with higher confidence
        #frame[:][:481] cause the bottom 200px is js my selection rectangle
        results = model(frame[:FRAME_WIDTH-SELECTION_AREA_WIDTH][:], conf=0.3)  # lower confidence threshold for better detection

        # process body detections
        for r in results:
            if hasattr(r, "keypoints") and r.keypoints is not None:
                keypoints = r.keypoints.xy.cpu().numpy()

                # THIS PART WAS SO DIFFICULT 
                for person_idx, kp in enumerate(keypoints):  # loop through all detected people with unique index
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
                        chin_y = int(nose_y + (nose_to_eye_distance * 2.5))  # adjust multiplier for better accuracy

                        # calculate head tilt angle
                        angle = np.degrees(np.arctan2(left_eye_y - right_eye_y, left_eye_x - right_eye_x))

                    except IndexError:
                        continue  

                    # more lenient ear position check
                    if left_ear_y > chin_y or right_ear_y > chin_y:
                        continue  

                    # more lenient chin position check
                    if chin_y < nose_y or chin_y - nose_y > 400:  # increased threshold
                        continue  

                    # more lenient ear spacing check
                    ear_distance = abs(right_ear_x - left_ear_x)
                    if ear_distance < 15 or ear_distance > 400:  # adjusted thresholds
                        continue  

                    # improved face width calculation
                    face_width = ear_distance
                    face_height = abs(chin_y - eye_mid_y) * 2.2  # making sure the crown fits just right

                    # size matters
                    filter_width = max(120, int(face_width * face_width_multipliers.get(current_face_filter, 1.3)))  # use filter-specific width multiplier
                    filter_height = max(140, int(face_height * face_height_multipliers.get(current_face_filter, 1.6)))  # use filter-specific height multiplier

                    # perfect placement
                    new_filter_x = nose_x - filter_width // 2
                    # adjust vertical position based on filter type
                    if current_face_filter == 2:  # birthday hat
                        # Position hat on top of head
                        head_top_y = eye_mid_y - (face_height * 0.5)  # Move higher to estimate top of head
                        new_filter_y = head_top_y - (filter_height * 0.7)  # Place hat directly on top of head
                        # Smoother rotation for hat
                        angle = angle * 0.8  # Reduce rotation intensity for more natural hat movement
                    else:
                        new_filter_y = nose_y - int(filter_height * 0.6)  # regular face filter position

                    # smooth transitions for each person independently
                    if person_idx in prev_filter_params_face:
                        prev_x, prev_y, prev_w, prev_h, prev_a = prev_filter_params_face[person_idx]
                        filter_x = int(0.3 * prev_x + 0.7 * new_filter_x)  
                        filter_y = int(0.3 * prev_y + 0.7 * new_filter_y)
                        filter_width = int(0.3 * prev_w + 0.7 * filter_width)
                        filter_height = int(0.3 * prev_h + 0.7 * filter_height)
                    else:
                        filter_x = new_filter_x
                        filter_y = new_filter_y

                    # store parameters for each person
                    prev_filter_params_face[person_idx] = (filter_x, filter_y, filter_width, filter_height, angle)

                    # boundary checks
                    filter_x = max(0, min(filter_x, frame.shape[1] - filter_width))
                    filter_y = max(0, min(filter_y, frame.shape[0] - filter_height))

                    

                    #WE ASSUME THIS PERSON IS REAL ATP
                    #we saw this fella 
                    last_seen.append(person_idx)
                    #add this fella into the dict if hes new
                    if (not person_idx in people): 
                        people[person_idx] = Person(0,0,0,0,0,0,0,0,0,0,0)
                    if (person_idx in people): 
                        people[person_idx].face_x = nose_x
                        people[person_idx].face_y = nose_y
                    #tell the rest of the code what the current filters are
                    current_face_filter = people[person_idx].face_image_it
                    current_body_filter = people[person_idx].body_image_it
                    current_hand_filter = people[person_idx].hand_image_it


                    # LEBRONIFICATION
                    if current_face_filter != 0:
                        frame = overlay_image(frame, pictures[current_face_filter].file, filter_x, filter_y, (filter_width, filter_height), angle)



                    # T-SHIRTIFICATION
                    if len(kp) >= 12:
                        # get shoulder points with better precision
                        left_shoulder = tuple(map(int, kp[5]))
                        right_shoulder = tuple(map(int, kp[6]))
                        # get hip points for better proportions
                        left_hip = tuple(map(int, kp[11]))
                        right_hip = tuple(map(int, kp[12]))

                        # fixed proportions and validation
                        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
                        torso_height = abs((left_hip[1] + right_hip[1])/2 - (left_shoulder[1] + right_shoulder[1])/2)
                        
                        # custom multipliers for each shirt
                        width_multiplier = shirt_width_multipliers.get(current_body_filter, 2.5)
                        height_multiplier = shirt_height_multipliers.get(current_body_filter, 2.0)
                        shirt_width = max(50, int(shoulder_width * width_multiplier))
                        shirt_height = max(50, int(torso_height * height_multiplier))

                        # better shirt positioning using midpoints
                        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) // 2
                        shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) // 2
                        shirt_x = int(shoulder_center_x - shirt_width // 2)
                        shirt_y = int(shoulder_center_y - shirt_height * shirt_offsets.get(current_body_filter, 0))  # use shirt-specific offset

                        # skip if dimensions are invalid
                        if shirt_width <= 0 or shirt_height <= 0:
                            continue

                        # better angle calculation using both shoulders
                        shoulder_angle = np.degrees(np.arctan2(right_shoulder[1] - left_shoulder[1],
                                                            right_shoulder[0] - left_shoulder[0]))
                        shirt_angle = 0  # keep shirt perfectly upright

                        # apply the overlay with adjusted parameters
                        if current_body_filter != 0:
                            frame = overlay_image(frame, pictures[current_body_filter].file, shirt_x, shirt_y,
                                            (shirt_width, shirt_height), shirt_angle)
                        
                        #store filter position
                        prev_filter_params_body[person_idx] = (shirt_x, shirt_y, shirt_width, shirt_height, shirt_angle)
                        
                        #update this guys body postition
                        if (person_idx in people): 
                            #print("updating :",  int(left_shoulder[0] + (shoulder_width//2)), " :", int(left_shoulder[1] + (torso_height//2)))
                            people[person_idx].body_x = int(left_shoulder[0] - (shoulder_width//2))
                            people[person_idx].body_y = int(left_shoulder[1] + (torso_height//2))
                    



                    # Handle hand filters (gauntlet) - Moved outside of shirt filter block
                    if len(kp) >= 16:
                        # Get wrist and elbow keypoints for better angle calculation
                        left_wrist = tuple(map(int, kp[9]))
                        right_wrist = tuple(map(int, kp[10]))
                        left_elbow = tuple(map(int, kp[7]))
                        right_elbow = tuple(map(int, kp[8]))

                        #update this guys wrist position
                        if (person_idx in people): 
                            people[person_idx].l_hand_x, people[person_idx].l_hand_y = left_wrist
                            people[person_idx].r_hand_x, people[person_idx].r_hand_y = right_wrist
                            #print("updating, 1:" ,left_wrist[0], " 2:",left_wrist[1])
                        
                        # Calculate hand dimensions and angles for both hands
                        for wrist, elbow in [(left_wrist, left_elbow), (right_wrist, right_elbow)]:
                            if all(coord > 0 for coord in wrist) and all(coord > 0 for coord in elbow):
                                # Calculate hand size based on body proportions
                                hand_size = int(shoulder_width * 0.6)  # Increased base size
                                hand_width = int(hand_size * hand_width_multipliers.get(current_hand_filter,0))
                                hand_height = int(hand_size * hand_height_multipliers.get(current_hand_filter,0))
                                
                                # Calculate angle between elbow and wrist
                                hand_angle = np.degrees(np.arctan2(wrist[1] - elbow[1], wrist[0] - elbow[0])) + 90  # Added 90 degrees for clockwise rotation
                                
                                # Create a copy of the gauntlet and flip it for right hand
                                gauntlet_overlay = pictures[current_hand_filter].file.copy()
                                if wrist == right_wrist:
                                    gauntlet_overlay = cv2.flip(gauntlet_overlay, 1)  # Horizontal flip for right hand
                                
                                # Adjust position to better align with hand
                                hand_x = wrist[0] - hand_width // 3  # Adjusted horizontal offset
                                hand_y = wrist[1] - hand_height // 1.5  # Increased vertical offset to position gauntlets higher
                                
                                # Apply the gauntlet overlay with rotation
                                if current_hand_filter != 0:
                                    print("im the problem")
                                    frame = overlay_image(frame, gauntlet_overlay,
                                                    hand_x, hand_y, (hand_width, hand_height), hand_angle)
                                if wrist == left_wrist:
                                    prev_filter_params_hands_l[person_idx] = (hand_x, hand_y, hand_width, hand_height, hand_angle)
                                else:
                                    prev_filter_params_hands_r[person_idx] = (hand_x, hand_y, hand_width, hand_height, hand_angle)
                            else:
                                print("should be dead though")
                                prev_filter_params_hands_l[person_idx] = (0,0,0,0,0)
                                prev_filter_params_hands_r[person_idx] = (0,0,0,0,0)
                    else:
                        print("should be dead though")
                        prev_filter_params_hands_l[person_idx] = (0,0,0,0,0)
                        prev_filter_params_hands_r[person_idx] = (0,0,0,0,0)
                    
         
    else: # find out the old posittion of filters and put them on
        #overlay old ones when not processing
        for person_idx in last_seen:
            if (people[person_idx].body_image_it != 0 and person_idx in prev_filter_params_body):
                filter_x, filter_y, filter_width, filter_height, angle = prev_filter_params_body[person_idx]
                if (filter_width != 0): #js a sanity check to make sure the values are initalised
                    frame = overlay_image(frame, pictures[people[person_idx].body_image_it].file, filter_x, filter_y, (filter_width, filter_height), angle)

            if (people[person_idx].face_image_it != 0  and person_idx in prev_filter_params_face):
                filter_x, filter_y, filter_width, filter_height, angle = prev_filter_params_face[person_idx]
                if (filter_width != 0):
                    frame = overlay_image(frame, pictures[people[person_idx].face_image_it].file, filter_x, filter_y, (filter_width, filter_height), angle)

            if (people[person_idx].hand_image_it != 0):
                if person_idx in prev_filter_params_hands_l:
                    filter_x, filter_y, filter_width, filter_height, angle = prev_filter_params_hands_l[person_idx]
                    if (filter_width > 0):
                        print("mememe")
                        frame = overlay_image(frame, pictures[people[person_idx].hand_image_it].file, filter_x, filter_y, (filter_width, filter_height), angle)
                if person_idx in prev_filter_params_hands_r:
                    filter_x, filter_y, filter_width, filter_height, angle = prev_filter_params_hands_r[person_idx]
                    if (filter_width > 0):
                        print(filter_width)
                        frame = overlay_image(frame, cv2.flip(pictures[people[person_idx].hand_image_it].file, 1), filter_x, filter_y, (filter_width, filter_height), angle)


    #create the selection area
    frame = cv2.rectangle(frame, (FRAME_WIDTH-SELECTION_AREA_WIDTH, 0), (FRAME_WIDTH-1,FRAME_HEIGHT-1), (128,128, 128),-1)
    frame = cv2.rectangle(frame, (FRAME_WIDTH-SELECTION_AREA_WIDTH, 0), (FRAME_WIDTH-1,FRAME_HEIGHT-1), (255,0, 0),5)
    #overlay the selections
    for i in range(len(selection_current)):
        frame = overlay_image(frame, pictures[i].file, selection_current[i].x, selection_current[i].y, (pictures[i].height, pictures[i].width), 0)

    #show aimbox
    if (drag_state):
        for person_idx in last_seen:
            #print(people[person_idx].body_x, people[person_idx].body_y)
            #print(PIC_TYPE[dragging_obj])
            if (PIC_TYPE[dragging_obj] == FACE or PIC_TYPE[dragging_obj] == CLEAR) and (people[person_idx].face_x > 0 and people[person_idx].face_y > 0):
                 if PIC_TYPE[dragging_obj] == FACE or people[person_idx].face_image_it != 0:
                    frame = cv2.circle(frame, (people[person_idx].face_x, people[person_idx].face_y), MARGIN_FACE,  (0, 225, 0), 5)
            
            if (PIC_TYPE[dragging_obj] == SHIRT or PIC_TYPE[dragging_obj] == CLEAR) and (people[person_idx].body_x > 0 and people[person_idx].body_y > 0):
                if PIC_TYPE[dragging_obj] == SHIRT or people[person_idx].body_image_it != 0:
                    #print(people[person_idx].body_x, people[person_idx].body_y)
                    frame = cv2.circle(frame, (people[person_idx].body_x, people[person_idx].body_y), MARGIN_BODY,  (0, 225, 0), 5)
            
            if (PIC_TYPE[dragging_obj] == HAND or PIC_TYPE[dragging_obj] == CLEAR):
                 if PIC_TYPE[dragging_obj] == HAND or people[person_idx].hand_image_it != 0:
                    if (people[person_idx].l_hand_x > 0 and people[person_idx].l_hand_y > 0):
                        frame = cv2.circle(frame, (people[person_idx].l_hand_x, people[person_idx].l_hand_y), MARGIN_HAND,  (0, 225, 0), 5)
                    if (people[person_idx].r_hand_x > 0 and people[person_idx].r_hand_y > 0):
                        frame = cv2.circle(frame, (people[person_idx].r_hand_x, people[person_idx].r_hand_y), MARGIN_HAND,  (0, 225, 0), 5)


                
    # showing FPS and controls info
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, 'hold q to quit', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("LeBron Filter + Full Tracking View (Multi-Person + Head Tilt Fixed)", frame)

    # keyboard controls
    key = cv2.waitKey(1)
    if  key == ord('q'): # press q to quit
        break
    
    

# thats game
cap.release()
cv2.destroyAllWindows()
