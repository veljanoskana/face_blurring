import cv2

# Otvoranje na slika
image = cv2.imread('images/haarcascade/ROI.jpg')

# Dobivanje na golemina na slika
height, width = image.shape[:2]

# Posakuvana golemina na pikselirana slika
w, h = (12, 12)

# Resize na vlezna slika vo posakuvana golemina
temp = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

# Vrakjanje vo prvobitna golemina na namalena slika
pixelated = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

cv2.imwrite("images/blur/pixelated.jpg", pixelated)
