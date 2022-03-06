import cv2
import numpy as np

# Odbiranje na slika na koja treba da i se najde centarot
circled_image = cv2.imread("images/blur/pixelated.jpg")
circled_gray = cv2.cvtColor(circled_image, cv2.COLOR_BGR2GRAY)

# Koristenje na momenti za da se najde centarot na slikata
# Vo X i Y se koordinatite na centarot
moment = cv2.moments(circled_gray)
X = int(moment["m10"]/moment["m00"])
Y = int(moment["m01"]/moment["m00"])

# Odbiranje na soodveten radius vo odnos na golemina na slika
(h, w) = circled_image.shape[:2]
radius = h/2

# Kreiranje na crna slika
mask = np.zeros(circled_image.shape[:2], dtype="uint8")
cv2.circle(mask, (X,Y), int(radius), 255, -1)
masked = cv2.bitwise_and(circled_image, circled_image, mask=mask)
cv2.imwrite("images/masks/masked.jpg", masked)

# Izlezni sliki
cv2.imshow("Maska", mask)
cv2.imshow("Maska na slika", masked)
cv2.waitKey(0)
cv2.destroyAllWindows()
