import cv2
import numpy as np
import math
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

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

# 윤곽선 검출
contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 이미지에 컨투어 그리기
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

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
    angle = math.atan2(dy, dx)

    # 데이터 추가
    data.append([center_x, center_y, radius, angle])

# DataFrame 생성
df = pd.DataFrame(data, columns=['X', 'Y', 'Radius', 'Angle'])

# 데이터 정렬
df = df.sort_values(by='Angle')

# 기준 벡터와 중심점 찾기
center_point = df.iloc[0]
closest_point = df.iloc[1]

# 기준 벡터 계산
basis_vector = ((closest_point['X'] - center_point['X']), (closest_point['Y'] - center_point['Y']))
scale = 10  # 기저 벡터의 스케일 조정

# 기준 벡터의 크기
basis_norm = np.linalg.norm(basis_vector)

# 선형 독립한 두 개의 기저 벡터 생성
basis_1 = basis_vector / basis_norm
basis_2 = np.array([-basis_1[1], basis_1[0]])

# 정방행렬 생성
basis_matrix = np.stack((basis_1, basis_2), axis=1)

# 기저 행렬의 역행렬 계산
basis_inv = np.linalg.inv(basis_matrix)

# 이미지에 점들 그리기
fig, ax = plt.subplots()
ax.scatter(df['X'], df['Y'], color='blue')  # 원본 데이터 점들

# 기저 벡터 그리기
origin = [center_point['X']], [center_point['Y']]
ax.quiver(*origin, basis_vector[0], basis_vector[1], color='green', angles='xy', scale_units='xy', scale=1)
ax.quiver(*origin, basis_1[0], basis_1[1], color='red', angles='xy', scale_units='xy', scale=1)
ax.quiver(*origin, basis_2[0], basis_2[1], color='red', angles='xy', scale_units='xy', scale=1)

# 중심점 표시
ax.scatter(center_point['X'], center_point['Y'], color='red')

# 변환된 좌표 계산
transformed_coordinates = np.dot(df[['X', 'Y']].values - center_point[['X', 'Y']].values, basis_inv.T)

# 변환된 좌표로 그리기
ax.scatter(transformed_coordinates[:, 0], transformed_coordinates[:, 1], color='purple')

plt.axis('equal')
plt.show()





# # 이미지 크기 조정
# scale = 10
# image_new = cv2.resize(image, (image.shape[1] * scale, image.shape[0] * scale))

# # 기준 벡터 그리기
# cv2.arrowedLine(image_new, (center_point['X'] * scale, center_point['Y'] * scale),
#                 ((center_point['X'] + basis_vector[0][0]) * scale, (center_point['Y'] + basis_vector[1][0]) * scale),
#                 (0, 0, 255), 2)

# # 중심점 그리기
# cv2.circle(image_new, (center_point['X'] * scale, center_point['Y'] * scale), 4, (0, 0, 255), -1)

# # 좌표 계산 및 이미지에 그리기
# for i in range(len(df)):
#     # 기저 벡터의 정수계수 reshape
#     integer_coefficients = df.iloc[i][['X', 'Y']].values.reshape((2, 1))

#     # 좌표 계산
#     coordinate = np.dot(basis_inv, integer_coefficients)

#     x = int(coordinate[0][0]) + center_point['X'] * scale
#     y = int(coordinate[1][0]) + center_point['Y'] * scale

#     cv2.circle(image_new, (x, y), 2, (0, 255, 0), -1)

# # 이미지 출력
# cv2.imshow('New Image', image_new)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
