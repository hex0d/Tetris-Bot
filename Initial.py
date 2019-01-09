import numpy as np
import cv2


from pynput.mouse import Listener
from PIL import ImageGrab


# doing the window already
'''def on_click(x, y, button, pressed):
    global topleft, botright
    if pressed:
        print("Press")
        topleft = (x,y)
    else:
        print('Release')
        listener.stop()

        botright = (x,y)
    if not pressed:
        # Stop listener
        return False


with Listener(on_click=on_click) as listener:
    listener.join()

print(topleft,botright)


while True:
    img = np.array(ImageGrab.grab(bbox=(topleft[0],topleft[1],botright[0],botright[1])))
    frame =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_read = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    cv2.imshow("Frame",frame)
    cv2.imshow("Read",img_read)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()'''

# working with a SS


# ## TDL - make the bot recognize tetromino via color


# ## Recognize first tetromino
# ## recognize next tetromino
# ## make self in-game matrix
# ## make decisions
# ## make inputs
# ##

tetrominos = ['l.png','i.png','z.png','j.png','s.png','o.png','t.png']
for mino in tetrominos:

    img = cv2.imread(mino)
    cv2.imshow('b', img)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    '''
    colours = [([ 89, 227, 205], [109, 247, 225],'light blue'),  # light blue
               ([104, 171 ,223], [124, 191, 243],'dark blue'),  # dark blue
               ([164, 227, 205], [184, 247, 225],'red'),  # red
               ([148, 185, 165], [168, 205, 185],'purple'),  # purple
               ([ 35, 244, 167], [ 55,   255, 187],'green'),  # green
               ([ 19, 243, 217], [ 23,   255, 237],'yellow'),  # yellow
               ([  10, 243, 217], [ 14,   255, 237],'orange')]  # orange'''


    colours = [([148, 153, 200], [168, 173, 220],'purple'),  # purple
               ([ 35, 202, 202], [ 55, 222, 222],'green'),  # green
               ([ 19, 208, 245], [ 25, 228, 255],'yellow'),  # yellow
               ([ 9, 208, 255], [ 15, 228, 255],'orange'),  # orange
               ([104, 171, 223], [124, 191, 243],'dark blue'),  # dark blue
               ([ 89, 194, 240], [109, 214, 255],'light blue'),  # light blue
               ([164, 194, 240], [184, 214, 255],'red')]  # red


    for (lower,upper,color) in colours:


        lower_a = np.array(lower)
        upper_a = np.array(upper)

        mask = cv2.inRange(img_hsv, lower_a, upper_a)
        cv2.imshow('a',mask)
        if np.count_nonzero(mask)>100:
            print(color)
            break

    cv2.destroyAllWindows()

# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#

# ret, mask = cv2.threshold(img_gray, 30, 255, cv2.THRESH_BINARY)
# cv2.imshow('a',img_hsv)
# template = cv2.imread('j_shape.png', 0)

# w, h = template.shape[::-1]

# res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
# threshold = 0.99
# loc = np.where(res >= threshold)

# for pt in zip(*loc[::-1]):
#     cv2.rectangle(img, pt,(pt[0]+w,pt[1]+h), (0,255,255), 2)

# cv2.imshow('TetrisGray',img_gray)

# cv2.imshow('detected',img_gray)

# cv2.imshow('TetrisThresh',mask)

# cv2.imshow('Tetris',img)

# cv2.imshow('res',res)

cv2.waitKey(0)
cv2.destroyAllWindows()