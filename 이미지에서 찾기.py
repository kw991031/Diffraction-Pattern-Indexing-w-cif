import cv2
import numpy as np
import math
import pandas as pd

# 이미지 불러오기
image = cv2.imread('tem_image2.jpg', 0)  # 흑백 이미지로 로드

# 이미지 전처리
ret, threshold = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)  # 임계값 적용하여 흑백 이진화

#원 그리기
x=113
y=101
cv2.circle(image, (x, y), 5, (255, 255, 255), 1)

# 결과 이미지 출력
cv2.imshow('Polar Coordinates', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 이미지 크기 설정
image_width = 800
image_height = 800

# 새로운 이미지 생성
image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

# 이미지 중심점 계산
center_x = image_width // 2
center_y = image_height // 2

# 극좌표를 가진 DataFrame 예시
df = pd.DataFrame({
    'R': [10, 20, 30, 40, 50],
    'Theta': [30, 60, 120, 200, 300]
})

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