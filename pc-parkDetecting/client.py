import socket
import struct
from PIL import Image
import io
import cv2
import numpy as np

# create an ipv4 (AF_INET) socket object using the tcp protocol (SOCK_STREAM)
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# connect the client
# client.connect((target, port))
#get_ip = os.popen('dns-sd -G  v4 raspberrypi7.local | grep  -E -o "(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"').read()
client.connect(('192.168.0.109', 50001))

while True:
    # receive the response data (4096 is recommended buffer size)
    s = client.recv(1)
    print(s)
    if s.decode('utf-8') == 'S':

        i = client.recv(3)
        print(i)
        l = client.recv(4)
        #print(l)
        length = int.from_bytes(l,byteorder='big')
        print(length)
        rem = length
        img = bytearray()
        while rem > 0:
            im = client.recv(rem)
            im = bytearray(im)
            img.extend(im)
            rem = rem-len(im)

        im = np.array(img)
        #print(type(im))
        print(im.shape)
        image = cv2.imdecode(im,3)
        cv2.imshow('test', image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
    else:
        client.close()








