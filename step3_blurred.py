import cv2
import numpy as np

# Otvoranje na slika koja treba da se zamagli
image = cv2.imread("images/haarcascade/ROI.jpg")

# Definiranje na shirochina, visina i faktor na zamagluvanje
(h, w) = image.shape[:2]

# Faktorot na zamagluvanje ja odreduva zamaglenosta na sliakta, shto pomal toa pogolema i zamaglenost
factor = 2.0

# Avtomatsko presmetuvanje na golemina na jadroto
kH = int(h/factor)
kW = int(w/factor)

# Sigurnost deka goleminata e neparna
if kW % 2 == 0:
	kW -= 1
# Sigurnost deka goleminata e neparna
if kH % 2 == 0:
	kH -= 1
gaussian = cv2.GaussianBlur(image, (kW,kH), 0)
cv2.imwrite("images/blur/gaussian.jpg", gaussian)
