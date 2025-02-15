import cv2
import numpy as np 

# Open the default camera
print("opening cam")
cam = cv2.VideoCapture(0)


size_y = 100
size_x = 200

update = True
mouseX = 550
mouseY = 500
drag_state = False
def change_pic_coord(event,x,y,flags,param):
    global mouseX,mouseY, drag_state, update, size_y, size_x
    if event == cv2.EVENT_LBUTTONDOWN:
        if drag_state: #put down
            drag_state = False
        elif abs(x - mouseX)-size_x//2-50 < 0 and abs(y - mouseY)-size_y//2-50: #pick up or nah
            drag_state = True
    elif event == cv2.EVENT_MOUSEMOVE and drag_state:
        mouseX,mouseY = x,y
        update = True
    print(event, x, y)


cv2.namedWindow('Camera')
cv2.setMouseCallback('Camera',change_pic_coord)

# Read logo and resize 
logo = cv2.imread('heart2.png') 
logo = cv2.resize(logo, (size_x, size_y)) 
  
# Create a mask of logo 
img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY) 
ret, mask = cv2.threshold(img2gray, 25, 255, cv2.THRESH_BINARY_INV)
img_mask = cv2.bitwise_not(mask)


while True:
    ret, frame = cam.read()
    if update:
        corner_x = mouseX-size_x//2
        corner_y = mouseY-size_y//2
        roi = frame[corner_y: size_y+corner_y, corner_x:size_x+corner_x] 
        roi_empty = cv2.bitwise_and(roi, roi, mask = img_mask)
        roi[np.where(mask)] = 0
        roi_empty += logo 

    # Display the captured frame
    cv2.imshow('Camera', frame)






    # Press 'q' to exit the loop
    if cv2.pollKey() == ord('q'):
        break

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()

