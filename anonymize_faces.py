# Вклучување на потребните библиотеки
import cv2
import numpy as np
import argparse


# Чекор 3: Функции кои извршуваат замаглување
def blurred(img, factor):
    (height, width) = img.shape[:2]
    kH = int(height / factor)
    kW = int(width // factor)
    if kH % 2 == 0:
        kH -= 1
    if kW % 2 == 0:
        kW -= 1
    blurred_image = cv2.GaussianBlur(img, (kW, kH), 0)
    return blurred_image


def pixelated(img):
    (height, width) = img.shape[:2]
    w, h = (12, 12)
    temp = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    pixelated_image = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
    return pixelated_image


# Модификација на ROI за естетски цели
def masking(roi, original):
    (height, width) = roi.shape[:2]
    mask = np.zeros((height, width), dtype="uint8")
    cv2.circle(original, (height // 2, width // 2), width // 2, 0, thickness=-1)
    cv2.circle(mask, (height // 2, width // 2), width // 2, 255, thickness=-1)
    masked = cv2.bitwise_and(roi, roi, mask=mask)
    circled = cv2.add(masked, original)
    cv2.imwrite("images/masks/circled.jpg", circled)
    return circled


# Конструкција на потребните аргументи
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Внесете патека до влезна слика")
ap.add_argument("-m", "--method", type=str, default="gaussian", choices=["gaussian", "pixelated"],
                help="Метод според кој сакате да биде анонимизирана сликата")
args = vars(ap.parse_args())

# Чекор 1: Создавање на детектор на лица преку Haar Cascade
detector = cv2.CascadeClassifier("venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")

# Отварање на слика од интерес и конвертирање на истата во црно-бела
# Поради особините на детекторот кој работи само со црно-бели слики
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_copy = image.copy()
image_height, image_width = image.shape[:2]

# Чекор 2: Пронаоѓање и одделување на регионот/регионите од интерес ROI
faces = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30),
                                  flags=cv2.CASCADE_SCALE_IMAGE)
print("Пронајдени се {0} лица!".format(len(faces)))
for (x, y, w, h) in faces:
    face = image[y:y + h, x:x + w]
    (fh, fw, channel) = face.shape

    if args["method"] == "gaussian":
        ann = blurred(face, factor=3.0)
    else:
        ann = pixelated(face)

    # Чекор 4: Замена на анонимизиран регион
    face = masking(ann, face)
    image[y:y + h, x: x + w] = face

# Приказ на резултат
cv2.imwrite("result.jpg", image)
result = np.hstack((image_copy, image))
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# python anonymize_faces.py -i images/test/bill.jpg -m gaussian
# python anonymize_faces.py -i images/test/multiple2.jpg -m pixelated
