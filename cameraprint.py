import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    print(f"Index {i}: {'OK' if cap.isOpened() else 'not found'}")
    cap.release()