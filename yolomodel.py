import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time

# load yolo model
model = YOLO("yolo11n-pose-better.pt")  

# hijack dat webcam 
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # making it small for speed
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)  # gotta go fast


prev_frame_time = 0
new_frame_time = 0
prev_filter_params = {}  # dictionary to store parameters for each person

# load all the images we need
lebron_face = cv2.imread("lebron.png", -1)
heart_face = cv2.imread("heart2.png", -1)
birthday_hat = cv2.imread("birthday_hat_transparent.png", -1)  # load the birthday hat image
tshirt = cv2.imread("sstshirt.png", -1)  # load the t-shirt image
blue_shirt = cv2.imread("bluesw.png", -1)  # load the blue shirt image
red_shirt = cv2.imread("redsw.png", -1)  # load the red shirt image
green_shirt = cv2.imread("greensw.png", -1)  # load the green shirt image
black_shirt = cv2.imread("blacksw.png", -1)  # load the black shirt image
yellow_shirt = cv2.imread("yellowsw.png", -1)  # load the yellow shirt image
jacket_torso = cv2.imread("sst_jacket_torso.png", -1)  # load the jacket torso image

# load gauntlet image
gauntlet = cv2.imread("transparent_gauntlet.png", -1)  # load the gauntlet image

# got gauntlet meh
if gauntlet is None or gauntlet.shape[0] == 0 or gauntlet.shape[1] == 0:
    raise FileNotFoundError("aint no transparent_gauntlet.png")

# got lebron meh
if lebron_face is None or lebron_face.shape[0] == 0 or lebron_face.shape[1] == 0:
    raise FileNotFoundError("aint no lebron.png")

# got heart meh
if heart_face is None or heart_face.shape[0] == 0 or heart_face.shape[1] == 0:
    raise FileNotFoundError("aint no heart2.png")

# got birthday hat meh
if birthday_hat is None or birthday_hat.shape[0] == 0 or birthday_hat.shape[1] == 0:
    raise FileNotFoundError("aint no birthday_hat_transparent.png")

# got tshirt meh
if tshirt is None or tshirt.shape[0] == 0 or tshirt.shape[1] == 0:
    raise FileNotFoundError("aint no sstshirt.png")

# got blue shirt meh
if blue_shirt is None or blue_shirt.shape[0] == 0 or blue_shirt.shape[1] == 0:
    raise FileNotFoundError("aint no bluesw.png")

# got red shirt meh
if red_shirt is None or red_shirt.shape[0] == 0 or red_shirt.shape[1] == 0:
    raise FileNotFoundError("aint no redsw.png")

# got green shirt meh
if green_shirt is None or green_shirt.shape[0] == 0 or green_shirt.shape[1] == 0:
    raise FileNotFoundError("aint no greensw.png")

# got black shirt meh
if black_shirt is None or black_shirt.shape[0] == 0 or black_shirt.shape[1] == 0:
    raise FileNotFoundError("aint no blacksw.png")

# got yellow shirt meh
if yellow_shirt is None or yellow_shirt.shape[0] == 0 or yellow_shirt.shape[1] == 0:
    raise FileNotFoundError("aint no yellowsw.png")

# got jacket torso meh
if jacket_torso is None or jacket_torso.shape[0] == 0 or jacket_torso.shape[1] == 0:
    raise FileNotFoundError("aint no sst_jacket_torso.png")



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

# filter management
face_filters = {0: None, 1: lebron_face, 2: heart_face, 3: birthday_hat}  # added birthday hat filter
face_width_multipliers = {1: 1.3, 2: 1.2, 3: 1.0}  # width multipliers for each face filter
face_height_multipliers = {1: 1.6, 2: 0.4, 3: 0.8}  # height multipliers for each face filter (adjusted for birthday hat)
shirt_filters = {0: None, 1: tshirt, 2: blue_shirt, 3: red_shirt, 4: green_shirt, 5: black_shirt, 6: yellow_shirt, 7: jacket_torso}  # Use jacket_torso as base image
shirt_offsets = {1: 0.3, 2: 0.3, 3: 0.2, 4: 0.3, 5: 0.3, 6: 0.3, 7: 0.3}  # adjusted offset for jacket_torso
shirt_width_multipliers = {1: 3, 2: 2.5, 3: 2.0, 4: 2.5, 5: 2.5, 6: 2.5, 7: 2.5}  # adjusted width multiplier for jacket_torso
shirt_height_multipliers = {1: 2.5, 2: 2, 3: 1.6, 4: 2, 5: 2, 6: 2, 7: 2}  # adjusted height multiplier for jacket_torso
hand_filters = {0: None, 1: gauntlet}  # add gauntlet filter for hands
hand_width_multipliers = {1: 2.5} 
hand_height_multipliers = {1: 2.5}  
current_face_filter = 1  # start with LeBron filter (1) or no filter (0)
current_shirt_filter = 1  # start with current t-shirt (1) or no filter (0)
current_hand_filter = 0  # start with no hand filter

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

    # run yolo inference on the frame with higher confidence
    results = model(frame, conf=0.3)  # lower confidence threshold for better detection

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
                if current_face_filter == 3:  # birthday hat
                    new_filter_y = eye_mid_y - int(filter_height * 5)  # position hat above the head
                else:
                    new_filter_y = nose_y - int(filter_height * 0.6)  # regular face filter position

                # smooth transitions for each person independently
                if person_idx in prev_filter_params:
                    prev_x, prev_y, prev_w, prev_h = prev_filter_params[person_idx]
                    filter_x = int(0.7 * prev_x + 0.3 * new_filter_x)  
                    filter_y = int(0.7 * prev_y + 0.3 * new_filter_y)
                    filter_width = int(0.7 * prev_w + 0.3 * filter_width)
                    filter_height = int(0.7 * prev_h + 0.3 * filter_height)
                else:
                    filter_x = new_filter_x
                    filter_y = new_filter_y

                # store parameters for each person
                prev_filter_params[person_idx] = (filter_x, filter_y, filter_width, filter_height)

                # boundary checks
                filter_x = max(0, min(filter_x, frame.shape[1] - filter_width))
                filter_y = max(0, min(filter_y, frame.shape[0] - filter_height))

                # LEBRONIFICATION
                if current_face_filter != 0 and face_filters[current_face_filter] is not None:
                    frame = overlay_image(frame, face_filters[current_face_filter], filter_x, filter_y, (filter_width, filter_height), angle)

                # T-SHIRTIFICATION
                if current_shirt_filter != 0 and shirt_filters[current_shirt_filter] is not None and len(kp) >= 12:
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
                    width_multiplier = shirt_width_multipliers.get(current_shirt_filter, 2.5)
                    height_multiplier = shirt_height_multipliers.get(current_shirt_filter, 2.0)
                    shirt_width = max(50, int(shoulder_width * width_multiplier))
                    shirt_height = max(50, int(torso_height * height_multiplier))

                    # better shirt positioning using midpoints
                    shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) // 2
                    shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) // 2
                    shirt_x = int(shoulder_center_x - shirt_width // 2)
                    shirt_y = int(shoulder_center_y - shirt_height * shirt_offsets[current_shirt_filter])  # use shirt-specific offset

                    # skip if dimensions are invalid
                    if shirt_width <= 0 or shirt_height <= 0:
                        continue

                    # better angle calculation using both shoulders
                    shoulder_angle = np.degrees(np.arctan2(right_shoulder[1] - left_shoulder[1],
                                                        right_shoulder[0] - left_shoulder[0]))
                    shirt_angle = 0  # keep shirt perfectly upright

                    # apply the overlay with adjusted parameters
                    frame = overlay_image(frame, shirt_filters[current_shirt_filter], shirt_x, shirt_y,
                                        (shirt_width, shirt_height), shirt_angle)
                
                # Handle hand filters (gauntlet) - Moved outside of shirt filter block
                if current_hand_filter != 0 and hand_filters[current_hand_filter] is not None and len(kp) >= 16:
                    # Get wrist and elbow keypoints for better angle calculation
                    left_wrist = tuple(map(int, kp[9]))
                    right_wrist = tuple(map(int, kp[10]))
                    left_elbow = tuple(map(int, kp[7]))
                    right_elbow = tuple(map(int, kp[8]))
                    
                    # Calculate hand dimensions and angles for both hands
                    for wrist, elbow in [(left_wrist, left_elbow), (right_wrist, right_elbow)]:
                        if all(coord > 0 for coord in wrist) and all(coord > 0 for coord in elbow):
                            # Calculate hand size based on body proportions
                            hand_size = int(shoulder_width * 0.6)  # Increased base size
                            hand_width = int(hand_size * hand_width_multipliers[current_hand_filter])
                            hand_height = int(hand_size * hand_height_multipliers[current_hand_filter])
                            
                            # Calculate angle between elbow and wrist
                            hand_angle = np.degrees(np.arctan2(wrist[1] - elbow[1], wrist[0] - elbow[0])) + 90  # Added 90 degrees for clockwise rotation
                            
                            # Create a copy of the gauntlet and flip it for right hand
                            gauntlet_overlay = hand_filters[current_hand_filter].copy()
                            if wrist == right_wrist:
                                gauntlet_overlay = cv2.flip(gauntlet_overlay, 1)  # Horizontal flip for right hand
                            
                            # Adjust position to better align with hand
                            hand_x = wrist[0] - hand_width // 3  # Adjusted horizontal offset
                            hand_y = wrist[1] - hand_height // 1.5  # Increased vertical offset to position gauntlets higher
                            
                            # Apply the gauntlet overlay with rotation
                            frame = overlay_image(frame, gauntlet_overlay,
                                                hand_x, hand_y, (hand_width, hand_height), hand_angle)


    

    # showing FPS and controls info
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, 'F: Switch Face Filter', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, 'S: Switch Shirt Filter', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, 'H: Switch Hand Filter', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, 'R: Remove All Filters', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("LeBron Filter + Full Tracking View (Multi-Person + Head Tilt Fixed)", frame)

    # keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):  # switch face filter
        current_face_filter = (current_face_filter + 1) % len(face_filters)
    elif key == ord('s'):  # switch shirt filter
        current_shirt_filter = (current_shirt_filter + 1) % len(shirt_filters)
    elif key == ord('h'):  # switch hand filter
        current_hand_filter = (current_hand_filter + 1) % len(hand_filters)
    elif key == ord('r'):  # remove all filters
        current_face_filter = 0
        current_shirt_filter = 0
        current_hand_filter = 0

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# thats game
cap.release()
cv2.destroyAllWindows()
