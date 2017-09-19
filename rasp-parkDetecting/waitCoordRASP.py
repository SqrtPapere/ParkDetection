import numpy as np
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

from timeit import default_timer as timer
from array import array
import socket
import os


def resize(mat, preferred_dimensions = (720,480)):
    original_dimensions = (mat.shape[0], mat.shape[1])
    if original_dimensions != (0,0):
        max_size_out = min(preferred_dimensions)
        max_size_in = max(original_dimensions)
        factor = max_size_out/max_size_in
        if factor < 1 :
            mat = cv2.resize(mat, dsize=None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
            # print("Resized")
    return mat

def mat_to_byte_array(mat, ext):
    #mat = resize(mat)
    # print(mat.shape)
    succes, img = cv2.imencode(ext, mat)
    bytedata = img.tostring()
    return succes, bytedata


#scatta foto 
camera = PiCamera()
time.sleep(0.5)
rawCapture = PiRGBArray(camera)
cc = camera.capture_continuous(rawCapture, format="bgr", use_video_port=True)
#image = rawCapture.array
image = next(cc).array




#open connection, send photo and wait
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    get_ip = os.popen('ifconfig | grep -Eo "inet (addr:)?([0-9]*\.){3}[0-9]*" | grep -Eo "([0-9]*\.){3}[0-9]*" | grep -v "127.0.0.1"').read()

    myip = get_ip.strip("\n")

    server_address = (myip, 50001)

    print(server_address)

    sock.bind(server_address)
    sock.listen(1)

    print("Listening")

    connection, client_address = sock.accept()
    print("Connected to " + str(client_address))

    success, encodedimg = mat_to_byte_array(image, ".jpg")

    if success:
            # print(str(len(encodedimg)))
            start = "S"
            type = "img"
            array = bytearray(start.encode('utf-8'))

            array.extend(type.encode('utf-8'))
            array.extend(len(encodedimg).to_bytes(4,'big'))
            array.extend(encodedimg)

            try:
                connection.sendall(array)
                # print("Frame sent")
            except:
                print("Connection Lost, reconnecting...")
                connection, client_address = sock.accept()
                print("Connected to " + str(client_address))
        #time.sleep(0.1)

#quando arriva parsa i risultati

	#data = connection.recv(16)
    s = connection.recv(1)
    print(s)
    if s.decode('utf-8') == 'S':

        i = connection.recv(3)
        print(i)
        l = connection.recv(4)
        #print(l)
        length = int.from_bytes(l, byteorder='big')
        print(length)
        rem = length
        coord = bytearray()
        while rem > 0:
            coo = connection.recv(rem)
            coo = bytearray(coo)
            coord.extend(coo)
            rem = rem-len(coo)

        scoord = coord.decode('utf-8')

    else:
        coonection.close()

    #crea coordinates.txt like (390, 223)(186, 73)(747, 236)(414, 73)
    if scoord is not "":
        f = open('coordinates.txt','w')
        f.write(scoord)
        f.close()
        print("coordinates received and stored")
        connection.close()
except KeyboardInterrupt:
    # Clean up the connection
    connection.close()
    print("Connection closed")
    # raise KeyboardInterrupt