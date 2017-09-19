# import the necessary packages
import cv2

import socket
import struct
from PIL import Image
import io
import numpy as np
import os

#start tcp connection with rasp and wait for the photo

print("Waiting image...")

# create an ipv4 (AF_INET) socket object using the tcp protocol (SOCK_STREAM)
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# connect the client
# client.connect((target, port))

# dns-sd -G  v4 raspberrypi7.local | grep  -E -o "(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
#get_ip = os.popen('dns-sd -G  v4 raspberrypi7.local | grep  -E -o "(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"').read()

client.connect(('192.168.0.109', 50001))

# receive the response data (4096 is recommended buffer size)
s = client.recv(1)
print(s)
if s.decode('utf-8') == 'S':

    i = client.recv(3)
    print(i)
    l = client.recv(4)
    # print(l)
    length = int.from_bytes(l, byteorder='big')
    print(length)
    rem = length
    img = bytearray()
    while rem > 0:
        im = client.recv(rem)
        im = bytearray(im)
        img.extend(im)
        rem = rem-len(im)

    im = np.array(img)
    # print(type(im))
    print(im.shape)
    image = cv2.imdecode(im,3)
else:
    client.close()

refPt = []
cropping = False


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append((x, y))
        cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False
        # draw a rectangle around the region of interest
        cv2.rectangle(image, refPt[len(refPt)-2], refPt[len(refPt)-1], (0, 255, 0), 2)
        cv2.imshow("image", image)


print("Selezionare un numero arbitrario di parking lot e premere il tasto c.")
print("\n")
print("In caso di errore premere il tasto r per resettare le selezioni.")

# load the image, clone it, and setup the mouse callback function

clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    # if the 'r' key is pressed, reset the cropping region
    if key == ord("r"):
        image = clone.copy()
        refPt = []

    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        break

if len(refPt)%2 is not 0:
    print("selezione non corretta, riprovare...")
elif len(refPt) >= 2:
    print(refPt)

    coord = ""
    for element in refPt:
        coord += (str(element))

    start = "S"
    type = "cmd"
    array = bytearray(start.encode('utf-8'))

    array.extend(type.encode('utf-8'))
    array.extend(len(coord).to_bytes(4,'big'))
    array.extend(coord.encode('utf-8'))

    try:
        client.sendall(array)
        print("Coordinates sent")
    except:
        print("Failed to send coordinates from pc to rasp...")

    cv2.waitKey(0) #wait fo a key to be pressed and send

# close all open windows
cv2.destroyAllWindows()
