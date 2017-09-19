#! /usr/bin/env python
"""Run a YOLO_v2 style detection model on test images."""
import colorsys
import imghdr
import os
import io
import random
import socket
import requests

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

from yolo.yad2k.models.keras_yolo import yolo_eval, yolo_head

import time
import timeit
import sys
import datetime
import logging

import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera

# vlc tcp/h264://dns-sd -G  v4 raspberrypi7.local | grep  -E -o "(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)":50001

time_url = 'http://api.timezonedb.com/v2/get-time-zone?key=HLAF2RCDDEWF&by=zone&zone=Europe/Rome&format=json'

def testRectIn(targX, targY, targH, targW, left, top, right, bottom):
    x_1 = max(targX, left)  # prendo i 4 punti, quelli di target e di box, in questo modo trovo le intersezioni
    y_1 = max(targY, top)  # poi faccio il confronto fra le due aree
    x_2 = min(targH + targX, right)
    y_2 = min(targW + targY, bottom)
    A_yolo = (right - left) * (bottom - top)
    A_inter = (x_2 - x_1) * (y_2 - y_1)  # area intersezione
    A_target = (targW) * (targH)  # area target
    if A_inter / A_target >= 0.3 and A_yolo / A_target < 3:  # altro coso da settare sperimentalmente
        return True
    return False


def mat_to_byte_array(mat, ext):
    # mat = resize(mat)
    # print(mat.shape)
    succes, img = cv2.imencode(ext, mat)
    bytedata = img.tostring()
    return succes, bytedata


def resize(mat, preferred_dimensions=(720, 480)):
    original_dimensions = (mat.shape[0], mat.shape[1])
    if original_dimensions != (0, 0):
        max_size_out = min(preferred_dimensions)
        max_size_in = max(original_dimensions)
        factor = max_size_out/max_size_in
        if factor < 1:
            mat = cv2.resize(mat, dsize=None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
            # print("Resized")
    return mat

try:

    model_path = "yolo/model_data/tiny-yolo-voc.h5"
    assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'
    anchors_path = "yolo/model_data/tiny-yolo-voc_anchors.txt"
    classes_path = "yolo/model_data/pascal_classes.txt"

    sess = K.get_session()
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

    yolo_model = load_model(model_path)

    # Verify model, anchors, and classes are compatible
    num_classes = len(class_names)
    num_anchors = len(anchors)
    model_output_channels = yolo_model.layers[-1].output_shape[-1]
    assert model_output_channels == num_anchors * (num_classes + 5), \
        'Mismatch between model and given anchor and class sizes. ' \
        'Specify matching anchors and classes with --anchors_path and ' \
        '--classes_path flags.'
    # print('{} model, anchors, and classes loaded.'.format(model_path))

    # Check if model is fully convolutional, assuming channel last order.
    model_image_size = yolo_model.layers[0].input_shape[1:3]
    is_fixed_size = model_image_size != (None, None)

    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.

    # Generate output tensor targets for filtered bounding boxes.
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2,))
    boxes, scores, classes = yolo_eval(yolo_outputs, input_image_shape, score_threshold=.3, iou_threshold=.5)

    logging.basicConfig(filename='carLog.log', level=logging.DEBUG)
    toFind = "car"

    # instaurazione server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    get_ip = os.popen(
        'ifconfig | grep -Eo "inet (addr:)?([0-9]*\.){3}[0-9]*" | grep -Eo "([0-9]*\.){3}[0-9]*" | grep -v "127.0.0.1"').read()

    myip = get_ip.strip("\n")

    server_address = (myip, 50001)

    print(server_address)

    sock.bind(server_address)
    sock.listen(1)

    print("Listening")

    connection, client_address = sock.accept()
    print("Connected to " + str(client_address))

    #fine impostazioni connessione

    # cap = cv2.VideoCapture("video.mkv")
    camera = PiCamera()
    # camera.resolution = (640, 480)
    # camera.framerate = 32
    rawCapture = PiRGBArray(camera)

    time.sleep(0.1)



    # read coordinates.txt

    coord = open('coordinates.txt', 'r')
    array = []
    for item in coord:
        array.append(item)
    coord.close()

    lines = array[0].split(')')
    line = ''.join(lines)
    lines = line.split('(')
    del lines[0]
    position = []
    for item in lines:
        a = item.split(',')
        position.append([int(a[0]), int(a[1])])

    # position = [[x,y], [x,y], [x,y], [x,y]]
    print("position = " + str(position))

    nTarget = int((len(position))/2)
    print("nTarget = " + str(nTarget))

    variables = []
    for i in range(nTarget):
        variables.append([False, False, 0])
    print(variables)

    target = []

    for l in range(0, len(position)-1, 2):
        cord1 = [position[l]]
        cord2 = [position[l+1]]
        target.append([cord1[0][0], cord1[0][1], cord2[0][0]-cord1[0][0], cord2[0][1]-cord1[0][1]])
    print("target = " + str(target))

    ccc = 0
    print(camera.resolution)

    for imagex in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

        print(ccc)
        ccc += 1
        # ret, im = cap.read()
        rawCapture.truncate(0)
        im = imagex.array
        image = Image.fromarray(im)

        carFound = []
        for i in range(0, nTarget):
            carFound.append([])

        if is_fixed_size:
            resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')
        else:
            # Due to skip connection + max pooling in YOLO_v2, inputs must have
            # width and height as multiples of 32.
            new_image_size = (image.width - (image.width % 32), image.height - (image.height % 32))
            resized_image = image.resize(new_image_size, Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = sess.run([boxes, scores, classes], feed_dict={yolo_model.input: image_data, input_image_shape: [image.size[1], image.size[0]], K.learning_phase(): 0})

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            data = [[top, left, bottom, right], predicted_class]

            for x in range(0, nTarget):
                if testRectIn(target[x][0], target[x][1], target[x][2], target[x][3], left, top, right, bottom) and data[1] == toFind:
                    carFound[x].append(data)

        for idx, it in enumerate(carFound):
            if len(it) is 0:
                variables[idx][0] = False
            else:
                variables[idx][0] = True

        font = ImageFont.truetype(
            font='yolo/font/FiraMono-Medium.otf',
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{}'.format(predicted_class)

            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for l in range(thickness):
                draw.rectangle(
                    [left + l, top + l, right - l, bottom - l],
                    outline=colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        image1 = np.array(image)
        for idr, rect in enumerate(target):
            cv2.rectangle(image1, (rect[0], rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (0, 170, 100), 2)
            cv2.rectangle(image1, (rect[0], rect[1] - 20), (rect[0] + 50, rect[1] - 2), (0, 170, 100), -1)
            cv2.putText(image1, 'Park'+str(idr), (rect[0], rect[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        #image1 = Image.fromarray(image1)

        for idy, obj in enumerate(variables):
            if obj[0] is not obj[1]:
                obj[2] += 1
                if obj[2] > 3:
                    # clock = time.strftime("%c")
                    r = requests.get(time_url)
                    data = r.json()
                    clock = data['formatted']
                    if obj[0]:
                        c_ar = "Car in Park" + str(idy) + " has arrived at: " + clock
                        print("\n" + c_ar)
                        cv2.imwrite('/home/pi/PROJECTPARKD/rasp-parkDetecting/CarMoved/' + c_ar + '.jpg', image1)
                    else:
                        c_l = "Car in Park" + str(idy) + " has left at: " + clock
                        print("\n" + c_l)
                        cv2.imwrite('/home/pi/PROJECTPARKD/rasp-parkDetecting/CarMoved/' + c_l + '.jpg', image1)
                    obj[1] = obj[0]
                    obj[2] = 0
            else:
                obj[2] = 0

        success, encodedimg = mat_to_byte_array(image1, ".jpg")

        if success:
            # print(str(len(encodedimg)))

            start = "S"
            type = "img"

            array = bytearray(start.encode('utf-8'))
            array.extend(type.encode('utf-8'))
            array.extend(len(encodedimg).to_bytes(4, 'big'))
            array.extend(encodedimg)

            try:

                connection.sendall(array)
                print(len(encodedimg))
                # print("im sent")
            except:
                print("Connection Lost, reconnecting...")
                connection, client_address = sock.accept()
                print("Connected to " + str(client_address))
                # time.sleep(0.1)


except KeyboardInterrupt:
    # Clean up the connection
    connection.close()
    print("Connection closed")



