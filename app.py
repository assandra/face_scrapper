import cv2
import sys
import csv
import os
from PIL import Image
import os, os.path

# Parameters to change
folder_name = 'cropped-mixed-god-dataset'
path = "/home/a/Projects/dggan-pytorch-2d-images/data/downloaded-images/"
stats_file_name = 'face_detection_mixed_god_dataset.csv'

# Loop through all image files of a dataset
status_array = []
for f in os.listdir(path):
    imagePath = os.path.join(path,f)
    print(imagePath)
    # Convert to grey scale for better results in facial recognition
    image = cv2.imread(imagePath)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces for loaded image
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        grey,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )

    # Draw rectangle around detected faces
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Save stats and cropped images for faces detected
    if (len(faces) > 0): 
        # Save cropped image
        croppedImagePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), folder_name, 'cropped-' + os.path.basename(imagePath))
        cv2.imwrite(croppedImagePath, image)
        # Save stats
        status_dict = {'face_detected': True, 'number_of_faces_detected': len(faces), 'original_image_path': imagePath, 'cropped_image_path':  croppedImagePath}
        status_array.append(status_dict)
     

# Write all stats to csv
with open(stats_file_name, mode='w') as csv_file:
    fieldnames = ['face_detected', 'number_of_faces_detected', 'original_image_path', 'cropped_image_path']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    for row in status_array:
        writer.writerow(row)
