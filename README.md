# Face Detection using Haar Cascades with OpenCV and Matplotlib
# NAME : DODLA SUSMITHA
# REG NO : 212224110016
# Aim
To write a Python program using OpenCV to perform the following image manipulations:
i) Extract ROI from an image.
ii) Perform face detection using Haar Cascades in static images.
iii) Perform eye detection in images.
iv) Perform face detection with label in real-time video from webcam.

# Software Required
Anaconda - Python 3.7 or above
OpenCV library (opencv-python)
Matplotlib library (matplotlib)
Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)
# Algorithm
# I) Load and Display Images
Step 1: Import necessary packages: numpy, cv2, matplotlib.pyplot
Step 2: Load grayscale images using cv2.imread() with flag 0
Step 3: Display images using plt.imshow() with cmap='gray'
# II) Load Haar Cascade Classifiers
Step 1: Load face and eye cascade XML files
# III) Perform Face Detection in Images
Step 1: Define a function detect_face() that copies the input image
Step 2: Use face_cascade.detectMultiScale() to detect faces
Step 3: Draw white rectangles around detected faces with thickness 10
Step 4: Return the processed image with rectangles
# IV) Perform Eye Detection in Images
Step 1: Define a function detect_eyes() that copies the input image
Step 2: Use eye_cascade.detectMultiScale() to detect eyes
Step 3: Draw white rectangles around detected eyes with thickness 10
Step 4: Return the processed image with rectangles
# V) Display Detection Results on Images
Step 1: Call detect_face() or detect_eyes() on loaded images
Step 2: Use plt.imshow() with cmap='gray' to display images with detected regions highlighted
# VI) Perform Face Detection on Real-Time Webcam Video
Step 1: Capture video from webcam using cv2.VideoCapture(0)
Step 2: Loop to continuously read frames from webcam
Step 3: Apply detect_face() function on each frame
Step 4: Display the video frame with rectangles around detected faces
Step 5: Exit loop and close windows when ESC key (key code 27) is pressed
Step 6: Release video capture and destroy all OpenCV windows
# Program
```
import cv2
import matplotlib.pyplot as plt
import numpy as np
```
```
w_glass = cv2.imread('image_02.png', cv2.IMREAD_GRAYSCALE)
wo_glass = cv2.imread('image_01.png', cv2.IMREAD_GRAYSCALE)
group = cv2.imread('image_03.png', cv2.IMREAD_GRAYSCALE)
```
```
w_glass1 = cv2.resize(w_glass, (1000, 1000))
wo_glass1 = cv2.resize(wo_glass, (1000, 1000)) 
group1 = cv2.resize(group, (1000, 1000))
```
```
plt.figure(figsize=(15,10))
plt.subplot(1,3,1);plt.imshow(w_glass1,cmap='gray');plt.title('With Glasses');plt.axis('off')
plt.subplot(1,3,2);plt.imshow(wo_glass1,cmap='gray');plt.title('Without Glasses');plt.axis('off')
plt.subplot(1,3,3);plt.imshow(group1,cmap='gray');plt.title('Group Image');plt.axis('off')
plt.show()
```
```
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def detect_and_display(image):
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 10)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()
```
```
import cv2
import urllib.request
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
urllib.request.urlretrieve(url, "face.xml")

face_cascade = cv2.CascadeClassifier("face.xml")

image = cv2.imread("image_02.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 5)

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 3)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Face Detection")
plt.axis("off")
plt.show()
```
```
from IPython.display import clear_output, display
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
```
```
import cv2
import urllib.request
import matplotlib.pyplot as plt

# Download Haar cascade files
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
    "face.xml")

urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml",
    "eye.xml")

face_cascade = cv2.CascadeClassifier("face.xml")
eye_cascade = cv2.CascadeClassifier("eye.xml")

# Load images
w_glass = cv2.imread("image_02.png")
wo_glass = cv2.imread("image_01.png")
group = cv2.imread("image_03.png")

def detect_eyes(image):
    result = image.copy()
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)

        roi = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi, 1.1, 5)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(result, (x+ex, y+ey),
                          (x+ex+ew, y+ey+eh), (0, 255, 0), 2)

    return result

for image, title in [
    (w_glass, "With Glasses - Eye Detection"),
    (wo_glass, "Without Glasses - Eye Detection"),
    (group, "Group - Eye Detection")
]:
    if image is not None:
        result = detect_eyes(image)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
        plt.show()
```
```
import cv2
import urllib.request
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
urllib.request.urlretrieve(url, "face.xml")

face_cascade = cv2.CascadeClassifier("face.xml")
cap = cv2.VideoCapture(0)

ret, frame = cap.read()

if ret:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title("Video Face Detection")
    plt.axis("off")
    plt.show()
else:
    print("No frame captured from camera.")

cap.release()
```
```
def new_detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    return frame
```
```
video_capture = cv2.VideoCapture(0)
captured_frame = None   

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("No frame captured from camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = new_detect(gray, frame)
    clear_output(wait=True)
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Video - Face & Eye Detection")
    display(plt.gcf())
    captured_frame = canvas.copy()  
    break
```
```
video_capture.release()
if captured_frame is not None and captured_frame.size > 0:
    cv2.imwrite('captured_face_eye.png', captured_frame)
    captured_image = cv2.imread('captured_face_eye.png', cv2.IMREAD_GRAYSCALE)
    plt.imshow(captured_image, cmap='gray')
    plt.title('Captured Face with Eye Detection')
    plt.axis('off')
    plt.show()
else:
    print("No valid frame to save.")
```
```
image = cv2.imread('image_04.png.png') 
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0) 

edges = cv2.Canny(blurred_image, 50, 150)  

plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis('off')
```
```
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

result_image = image.copy() 
for contour in contours:
    if cv2.contourArea(contour) > 50: 
        x, y, w, h = cv2.boundingRect(contour)  
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  


plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.title("Handwriting Detection")
plt.axis('off')
```
# OUTPUT
<img width="1182" height="383" alt="download" src="https://github.com/user-attachments/assets/464845ef-aa04-4464-9bf9-f8b013195ed6" />

<img width="404" height="410" alt="download" src="https://github.com/user-attachments/assets/a260fd0e-6370-406f-a5b2-cd76a664e290" />

<img width="299" height="410" alt="download" src="https://github.com/user-attachments/assets/e2d13a0a-6356-4570-908e-f4f182556528" />

<img width="515" height="347" alt="download" src="https://github.com/user-attachments/assets/1967b011-7228-4b64-864d-c435cc6b36cf" />

<img width="512" height="410" alt="download" src="https://github.com/user-attachments/assets/a6473d80-8ec1-4f84-8280-19bf257e9314" />

<img width="512" height="410" alt="download" src="https://github.com/user-attachments/assets/9a98e614-9c6d-41c1-8a36-52a5cf20cd89" />

<img width="512" height="410" alt="download" src="https://github.com/user-attachments/assets/a3698b95-e06e-4ea7-9843-c4ac6826380a" />

<img width="515" height="266" alt="download" src="https://github.com/user-attachments/assets/6d72b64a-d21b-470a-aaf6-85ace7a4ee99" />

<img width="515" height="266" alt="download" src="https://github.com/user-attachments/assets/4e329923-cce0-440b-804d-af75f9ed5b78" />

