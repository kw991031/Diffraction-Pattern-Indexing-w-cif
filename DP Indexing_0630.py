import cv2
import numpy as np
import math
import pandas as pd
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

# # 면적이 0인 컨투어는 건너뜀
# filtered_contours = []
# for contour in contours:
#     if cv2.contourArea(contour) == 0:
#         continue
#     filtered_contours.append(contour)

# 극좌표 계산 및 DataFrame 생성
data = []
for contour in contours:
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
imagebefore=image

df_new = df[df['Theta'] <= 180].copy()
df_new = df_new.sort_values('R')


# # "r" 값에 따라 데이터프레임 정렬
# df_sorted = df.sort_values(by='R')

# # 비슷한 "r" 값들을 묶어주는 그룹 식별
# groups = []
# current_group = [df_sorted.index[0]]

# for i in range(1, len(df_sorted)):
#     if abs(df_sorted['R'].iloc[i] - df_sorted['R'].iloc[i-1]) < 1:  # "r" 값의 차이가 1 이하인 경우 같은 그룹으로 묶음
#         current_group.append(df_sorted.index[i])
#     else:
#         groups.append(current_group)
#         current_group = [df_sorted.index[i]]
# groups.append(current_group)

# # 각 그룹에서 가장 위에 있는 값 하나씩만을 남기는 새로운 데이터프레임 생성
# data = []

# for group in groups:
#     top_index = group[0]  # 각 그룹의 첫 번째 인덱스를 선택하여 가장 위에 있는 값을 나타냄
#     data.append(df.loc[top_index].to_dict())

# # 결과 출력
# df_new = pd.DataFrame(data)
# print(df_new)

#기준이 되는 x와 y 좌표
center_x = df_new['X'].iloc[0]
center_y = df_new['Y'].iloc[0]

# 중심 좌표를 기준으로 x와 y 값 계산
df_new['x_relative'] = df_new['X'] - center_x
df_new['y_relative'] = df_new['Y'] - center_y

# 중심 좌표를 기준으로 x와 y 값 계산 (df에서도 똑같은 과정 반복)
df['x_relative'] = df['X'] - center_x
df['y_relative'] = df['Y'] - center_y

print(df_new)

# Define the basis vectors
basis = np.array([[df_new['x_relative'].iloc[1], df_new['y_relative'].iloc[1]],
                  [df_new['x_relative'].iloc[2], df_new['y_relative'].iloc[2]]])

print("basis:", basis)

# Calculate the basis vector's inverse
basis_inv = np.linalg.inv(basis)

# Add 'x_coefficient' and 'y_coefficient' columns to the DataFrame
df[['x_coefficient', 'y_coefficient']] = df.apply(lambda row: pd.Series(np.round(np.dot(basis_inv, np.array([row['x_relative'], row['y_relative']]))).astype(int)), axis=1)

print (df)
# Set the image size
image_width = image.shape[1]
image_height = image.shape[0]

# Create a blank image
new_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

# Calculate the center point of the image
center_x = image_width // 2
center_y = image_height // 2

# Draw basis vectors on the image
pt1 = (center_x, center_y)
pt2 = (center_x + int(basis[0, 0]), center_y + int(basis[0, 1]))
pt3 = (center_x + int(basis[1, 0]), center_y + int(basis[1, 1]))
cv2.arrowedLine(new_image, pt1, pt2, (0, 0, 255), 2)
cv2.arrowedLine(new_image, pt1, pt3, (0, 0, 255), 2)

# Get the points from the DataFrame
points = df[['x_coefficient', 'y_coefficient']].values

# 이미지에 점 그리기
for i, point in enumerate(points):
    x = int(center_x + point[0] * basis[0, 0] + point[1] * basis[1, 0])
    y = int(center_y + point[0] * basis[0, 1] + point[1] * basis[1, 1])
    cv2.circle(new_image, (x, y), 2, (0, 255, 0), -1)

# Display the image
cv2.imshow('New Image', new_image)
cv2.imshow('Image', imagebefore)
cv2.waitKey(0)
cv2.destroyAllWindows()