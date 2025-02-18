import cv2
import numpy as np 
import math

# Open the default camera
print("opening cam")
cam = cv2.VideoCapture(0)
WIDTH  = cam.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
HEIGHT = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

#sets height of the picture
size_y = 100
size_x = 200
#points to snap to
snapTo = [[550,500], [300,150],[650,150], [1100,150]]
#first postion the picture should go
mouseX = snapTo[0][0]
mouseY = snapTo[0][1]

#mouse click handler
update = True
set_down = False
drag_state = False
def change_pic_coord(event,x,y,flags,param):
    global mouseX,mouseY, drag_state, update, size_y, size_x, set_down
    if event == cv2.EVENT_LBUTTONDOWN:
         #put down the picture
        if drag_state:
            drag_state = False
            set_down = True
        #decide if the mouse click is close enuough to the picture to be clicked
        elif abs(x - mouseX)-size_x//2-50 < 0 and abs(y - mouseY)-size_y//2-50 < 0: 
            drag_state = True
            mouseX,mouseY = x,y
            update = True
    #when dragin update position
    elif event == cv2.EVENT_MOUSEMOVE and drag_state:
        mouseX,mouseY = x,y
        update = True
    print(event, x, y, update, mouseX, mouseY)


cv2.namedWindow('Camera')
cv2.setMouseCallback('Camera',change_pic_coord)

# Read logo and resize 
logo = cv2.imread('heart2.png') 
logo = cv2.resize(logo, (size_x, size_y)) 
  
# Create a mask of logo 
img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY) 
ret, mask = cv2.threshold(img2gray, 25, 255, cv2.THRESH_BINARY_INV)
img_mask = cv2.bitwise_not(mask)


corner_x, corner_y = 0, 0
while True:
    ret, frame = cam.read()
    if update:
        if set_down:
        #find closest snap to
            best = snapTo[0]
            #use ptythagorous to find distance to each snap
            dist = math.sqrt(math.pow(snapTo[0][0] - mouseX, 2) + math.pow(snapTo[0][1] - mouseY, 2))
            for pair in snapTo:
                tmp = math.sqrt(math.pow(pair[0] - mouseX, 2) + math.pow(pair[1] - mouseY, 2))
                if tmp < dist:
                    dist = tmp
                    best = pair
            #convert from middle coordinate to a corner cood for picture moving
            corner_x = best[0]-size_x//2
            corner_y = best[1]-size_y//2
            mouseX = best[0]
            mouseY = best[1]
            set_down = False
            update = False
        else:
            corner_x = mouseX-size_x//2
            corner_y = mouseY-size_y//2
    #edge detection to prevent picture exiting the fram and crashing shit
    if corner_x < 0:
        corner_x = 0
    if corner_y < 0:
        corner_y = 0
    if (corner_x + size_x) > WIDTH:
        corner_x = int(WIDTH-size_x)
    if (corner_y + size_y) > HEIGHT:
        corner_y = int(HEIGHT-size_y)

    #update the picture
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

