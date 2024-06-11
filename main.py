#Importing Libraries
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os

#Library for sending SMS messages
from twilio.rest import Client

#Library to get current location
import geocoder
from geopy.geocoders import Nominatim

from gpiozero import Buzzer, LED, Button
from time import sleep

bz-Buzzer(3)
#led-LED(16)
button-Button(24)
bz.off()
#led.off()

# Variable Declaration
global eyeCounter
global yawnCounter
global alarm_status
global alarm_status2

eyeCounter = 0
yawnCounter = 0

# Function to find current location and send alert via SMS
def sms():
    g = geocoder.ip('me')  # Finding location using IP address
    latitude = str(g.latlng[0])
    longitude = str(g.latlng[1])
    latitude = str(19.107600)
    longitude = str(72.836962)

    geolocator = Nominatim(user_agent="geoapiExercises")
    location = geolocator.geocode(latitude + "," + longitude)
    location_str = str(location)
    mobile_no = 9594996010

    msgbody = (f"Alert! The Driver Sanika Shete is heavily sleepy and is prone to accident.\n"
               f"Car Number: MH02BP2439, Location: {location_str}. Please take necessary action to prevent any serious accidents. "
               f"Click to view Map : https://www.google.com/maps/search/{latitude},{longitude}. Click to call : {mobile_no}")

    account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
    auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
    client = Client(account_sid, auth_token)

    message = client.api.account.messages.create(
        to="+919594996010",
        from_="+19794757815",
        body=msgbody
    )

    # print("SMS Sent")

# Function to control Buzzer and Push Button
def alert():
    bz.on()
    # led.blink()

    def things_off():
        bz.off()
        # led.off()

    button.when_pressed = things_off

# Alarm Function
def alarm():
    if alarm_status:
        print("Drowsiness Alert! Drowsy eyes Detected.")
        alert()

    if not alarm_status and alarm_status2:
        print("Drowsiness Alert! Driver is Yawning.")
        alert()


# Calculating Eye Aspect Ratio
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear


# Extracting eye landmarks and calculating EAR
def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)


# Extracting lip landmarks and calculating Mouth Aspect Ratio
def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance


# Initializing the camera
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
args = vars(ap.parse_args())


# Setting Threshold and Status variables
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 38
YAWN_THRESH = 15
alarm_status = False
alarm_status2 = False
COUNTER = 0
YAWNCOUNTER = 0
YAWN_CONSEC_FRAME = 20

print("Loading the Predictor and Detector...")

# Detecting Face from Video Frames
detector = dlib.get_frontal_face_detector()

# Detecting Facial Landmarks
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

print("Starting the Video Stream...")
vs = VideoStream(usePiCamera=True).start()
time.sleep(1.0)

while True:
    # Reading Video Stream
    frame = vs.read()
    frame = imutils.resize(frame, width=250)

    # Grayscale Conversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye[1]
        rightEye = eye[2]

        distance = lip_distance(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        # Detecting Drowsy Eyes
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                eyeCounter += 1
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if not alarm_status:
                    alarm_status = True
                    # Calling the Alarm Function
                    alarm()

                if eyeCounter > 4:
                    # Calling the SMS Function
                    sms()
                    eyeCounter = 0
        else:
            COUNTER = 0
            alarm_status = False

        # Detecting Driver Yawn
        if distance > YAWN_THRESH:
            yawnCounter += 1
            cv2.putText(frame, "YAWN ALERT", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if yawnCounter > 10:
                YAWNCOUNTER += 1
                if YAWNCOUNTER >= 3:
                    if not alarm_status2:
                        alarm_status2 = True
                        # Calling the Alarm Function
                        alarm()
                        # Calling the SMS Function
                        sms()
                    yawnCounter = 0
                    YAWNCOUNTER = 0
        else:
            alarm_status2 = False
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Stop the Program on pressing Q
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
