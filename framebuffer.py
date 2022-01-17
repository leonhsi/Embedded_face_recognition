import face_recognition
import cv2
import numpy as np
import os
import time 

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
end = 0
video_capture = cv2.VideoCapture(2)

obama_image = face_recognition.load_image_file("si.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

biden_image = face_recognition.load_image_file("lin.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]


known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding
]
known_face_names = [
    "si",
    "lin"
]


face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

buf = open('/dev/fb0', 'wb+')
faces = ''
startt = input('press s to start : ')

while startt == 's':
    start = time.time();
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.48)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    for face_landmarks in face_landmarks_list:
        if(len(face_landmarks['right_eye']) > 0):
            #for facial_feature in face_landmarks.keys():
            points = np.array(face_landmarks['right_eye'], np.int32) * 4
            cv2.polylines(frame, pts = [points], isClosed = True ,color = (255, 255, 255))
        if(len(face_landmarks['left_eye']) > 0):
            points = np.array(face_landmarks['left_eye'], np.int32) * 4
            cv2.polylines(frame, pts = [points], isClosed = True ,color = (255, 255, 255))


    if(len(face_names) > 0 and end == 0) :
        end = time.time()
        faces = str(((end - start)*1000)) + ": " + ', '.join(face_names)
    font = cv2.FONT_HERSHEY_DUPLEX    
    cv2.putText(frame, faces, (30, 30), font, 1.0, (255, 255, 255), 1)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGR565)
    frame = cv2.resize(frame, (640, 480))
    
    buf.seek(os.SEEK_SET)
    buf.write(frame)

# Release handle to the webcam
buf.close()
video_capture.release()
cv2.destroyAllWindows()
