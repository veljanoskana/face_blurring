import argparse
import cv2
import numpy as np

# Konstrukcija na argument parser i parsiranje na argumentot za slika
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True)
args=vars(ap.parse_args())

# Sozdavanje haar cascade detektor
detector = cv2.CascadeClassifier("venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")

# Zemanje na slika i pretvoranje vo crno bela
image = cv2.imread(args["image"])
image_backup = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Dobivanje na koordinatite koi go odbelezhuvaat likot
rectangles = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

# Iscrtuvanje na pravoagolnik okolu dobienite koordinati
for (x, y, w, h) in rectangles:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    ROI = image_backup[y:y + h, x:x + w]
    cv2.imshow("Rectangle", image)
    cv2.imshow("ROI", ROI)
    cv2.imwrite("rectangle.jpg", image)
    cv2.imwrite("roi.jpg", ROI)

# Zachuvuvanje rezultat od Haar Cascade detektor na lik
# cv2.imwrite("images/haarcascade/haarcascade.jpg", image)

# Oddeluvanje na detektiraniot del od slikata

# cv2.imwrite("images/haarcascade/ROI.jpg", ROI)










