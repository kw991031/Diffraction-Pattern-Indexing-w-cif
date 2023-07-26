import cv2
import numpy as np
import math
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import cv2

import tkinter as tk
from tkinter import filedialog
import os

# Tkinter 창 생성
root = tk.Tk()
root.withdraw()

# 파일 브라우저 열기
file_path = filedialog.askopenfilename()

# 파일명 추출
selected_filename = file_path.split('/')[-1]

print("Selected File:", selected_filename)


# 이미지 파일 읽기
image = cv2.imread(selected_filename, 0)


# 이미지 전처리
ret, threshold = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)  # 임계값 적용하여 흑백 이진화

cv2.imshow('any',threshold)
# 윤곽선 검출
contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 이미지에 컨투어 그리기
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
cv2.imshow('any',image)

# 이미지 중심점 계산
center_x = image.shape[1] // 2
center_y = image.shape[0] // 2

# 가장 큰 컨투어 찾기
largest_contour = max(contours, key=cv2.contourArea)

# 가장 큰 컨투어의 중심 좌표 계산
M = cv2.moments(largest_contour)
largest_center_x = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
largest_center_y = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0

# 면적이 0인 컨투어는 건너뜀
filtered_contours = []
for contour in contours:
    if cv2.contourArea(contour) == 0:
        continue
    filtered_contours.append(contour)

# 극좌표 계산 및 DataFrame 생성
data = []
for contour in filtered_contours:
    # 윤곽선 중심 좌표 계산
    M = cv2.moments(contour)
    center_x = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
    center_y = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0

    # 극좌표 계산
    dx = center_x - largest_center_x
    dy = center_y - largest_center_y
    radius = math.sqrt(dx ** 2 + dy ** 2)
    theta = math.atan2(dy, dx)

    # 각도를 0~360 범위로 변환
    if theta < 0:
        theta += 2 * math.pi

    data.append({'X': center_x, 'Y': center_y, 'R': radius, 'Theta': math.degrees(theta)})

# DataFrame 생성
df = pd.DataFrame(data)
df = df.sort_values('R')
NearestPointD = df['R'].iloc[1]
print(NearestPointD)
df['relativeR'] = df['R'] / NearestPointD

# 결과 출력
print(df)

#출력된 data frame을 기반으로 새로운 이미지 생성 (Simulated SAED)
# 이미지 크기 설정
image_width = 800
image_height = 800

# 새로운 이미지 생성
image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

# 이미지 중심점 계산
center_x = image_width // 2
center_y = image_height // 2

# 이미지에 극좌표 플롯
for index, row in df.iterrows():
    radius = row['R']
    theta = math.radians(row['Theta'])

    x = int(radius * math.cos(theta)) + center_x
    y = int(radius * math.sin(theta)) + center_y

    cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

# 이미지에 중심점 표시
cv2.circle(image, (center_x, center_y), 3, (0, 0, 255), -1)

# 이미지 출력
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()