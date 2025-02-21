import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time

# load yolo model
model = YOLO("yolo11n-pose-better.pt")  

# hijack dat mufuckin webcam bitchhhhh
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # making it small for speed
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)  # gotta go fast

# some optimisation shit that chatgpt suggested
prev_frame_time = 0
new_frame_time = 0
prev_filter_params = {}  # Dictionary to store parameters for each person

# load all the images we need
lebron_face = cv2.imread("lebron.png", -1)
tshirt = cv2.imread("sstshirt.png", -1)  # load the t-shirt image
blue_shirt = cv2.imread("bluesw.png", -1)  # load the blue shirt image

# got lebron meh
if lebron_face is None or lebron_face.shape[0] == 0 or lebron_face.shape[1] == 0:
    raise FileNotFoundError("aint no lebron.png")

# got tshirt meh
if tshirt is None or tshirt.shape[0] == 0 or tshirt.shape[1] == 0:
    raise FileNotFoundError("aint no sstshirt.png")

# got blue shirt meh
if blue_shirt is None or blue_shirt.shape[0] == 0 or blue_shirt.shape[1] == 0:
    raise FileNotFoundError("aint no bluesw.png")

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

# Filter management
face_filters = {0: None, 1: lebron_face}  # Add more face filters here
shirt_filters = {0: None, 1: tshirt, 2: blue_shirt}  # Added blue shirt as option 2
current_face_filter = 1  # Start with LeBron filter (1) or no filter (0)
current_shirt_filter = 1  # Start with current t-shirt (1) or no filter (0)

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

    # run yolo inference on the frame with higher confidence
    results = model(frame, conf=0.3)  # lower confidence threshold for better detection

    # Process body detections
    for r in results:
        if hasattr(r, "keypoints") and r.keypoints is not None:
            keypoints = r.keypoints.xy.cpu().numpy()

            # THIS PART WAS SO FUCKING DIFFICULT WTF BRO

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
                    chin_y = int(nose_y + (nose_to_eye_distance * 2.5))  # Adjust multiplier for better accuracy

                    # calculate head tilt angle
                    angle = np.degrees(np.arctan2(left_eye_y - right_eye_y, left_eye_x - right_eye_x))

                except IndexError:
                    continue  

                # more lenient ear position check
                if left_ear_y > chin_y or right_ear_y > chin_y:
                    continue  

                # more lenient chin position check
                if chin_y < nose_y or chin_y - nose_y > 400:  # Increased threshold
                    continue  

                # more lenient ear spacing check
                ear_distance = abs(right_ear_x - left_ear_x)
                if ear_distance < 15 or ear_distance > 400:  # Adjusted thresholds
                    continue  

                # improved face width calculation
                face_width = ear_distance
                face_height = abs(chin_y - eye_mid_y) * 2.2  # making sure the crown fits just right

                # size matters
                filter_width = max(120, int(face_width * 1.3))  # make room for the beard
                filter_height = max(140, int(face_height * 1.6))  # tall head

                # perfect placement
                new_filter_x = nose_x - filter_width // 2
                new_filter_y = nose_y - int(filter_height * 0.6) 

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
                if current_shirt_filter != 0 and shirt_filters[current_shirt_filter] is not None and len(kp) >= 12:  # Make sure we have enough keypoints for shoulders and hips
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
                        continue

                    # Improved angle calculation using both shoulders
                    shoulder_angle = np.degrees(np.arctan2(right_shoulder[1] - left_shoulder[1],
                                                        right_shoulder[0] - left_shoulder[0]))
                    shirt_angle = 0  # Keep shirt perfectly upright

                    # Apply the overlay with adjusted parameters
                    frame = overlay_image(frame, shirt_filters[current_shirt_filter], shirt_x, shirt_y,
                                        (shirt_width, shirt_height), shirt_angle)

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

                # Draw chin point and its connections (keeping face structure visualization)
                cv2.circle(frame, (chin_x, chin_y), 5, (0, 255, 0), -1)
                cv2.line(frame, (left_ear_x, left_ear_y), (chin_x, chin_y), (255, 0, 0), 2)
                cv2.line(frame, (right_ear_x, right_ear_y), (chin_x, chin_y), (255, 0, 0), 2)

    # Filter state is managed outside the loop to maintain proper switching

    # showing FPS and controls info
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, 'F: Switch Face Filter', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, 'S: Switch Shirt Filter', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, 'R: Remove All Filters', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("LeBron Filter + Full Tracking View (Multi-Person + Head Tilt Fixed)", frame)

    # keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):  # Switch face filter
        current_face_filter = (current_face_filter + 1) % len(face_filters)
    elif key == ord('s'):  # Switch shirt filter
        current_shirt_filter = (current_shirt_filter + 1) % len(shirt_filters)
    elif key == ord('r'):  # Remove all filters
        current_face_filter = 0
        current_shirt_filter = 0

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# thats game
cap.release()
cv2.destroyAllWindows()
