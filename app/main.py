import requests
import cv2
import numpy as np

IMG_SOURCE_URL = "http://server-khac/anh.jpg"  # Thay bằng URL thực tế

def fetch_and_show():
    resp = requests.get(IMG_SOURCE_URL)
    img_arr = np.frombuffer(resp.content, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if img is not None:
        cv2.imshow("Received Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Không nhận được ảnh hợp lệ.")

if __name__ == "__main__":
    fetch_and_show()