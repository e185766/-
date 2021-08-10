import cv2
from pathlib import Path
import time
import os
import psutil

IMAGE_PATH = "/Users/hinoueyuuya/大学/４年前期/並列分散処理/最終課題/man"

def mosaic(src, ratio=0.1):
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

def mosaic_area(src, x, y, width, height, ratio=0.1):
    dst = src.copy()
    dst[y:y + height, x:x + width] = mosaic(dst[y:y + height, x:x + width], ratio)
    return dst

def main():
    """ 100枚の画像から、100枚のモザイク処理した画像を生成する """
    files = Path(IMAGE_PATH + "/original/").glob("*")
    for file in files:
        src = cv2.imread(str(file))
        cascade_path = '/Users/hinoueyuuya/opt/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml'
        cascade = cv2.CascadeClassifier(cascade_path)
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(src_gray)
        for x, y, w, h in faces:
            dst = mosaic_area(src, x, y, w, h)
        cv2.imwrite(IMAGE_PATH + "/mosaic/" + file.name, dst)

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    mem = psutil.virtual_memory() 
    print(mem.percent)
    cpu = psutil.cpu_percent(interval=1)
    print(cpu)
    print("Finished in {} seconds.".format(end-start))