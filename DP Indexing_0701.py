import cv2
import numpy as np
import pandas as pd

# 이미지 파일 경로
image_path = 'LNMO SAED_crop.jpg'

# 이미지 로드
image = cv2.imread(image_path)

# 이미지 전처리 작업 (예시: 그레이스케일 변환)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 점 검출을 위한 작업 (예시: Canny 에지 검출)
edges = cv2.Canny(gray, 100, 200)

# 점들의 위치 검출 (예시: Hough 변환을 이용한 원 검출)
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=50, param2=30, minRadius=0, maxRadius=0)

# DataFrame 생성
df = pd.DataFrame(columns=['x', 'y'])

# 점들의 좌표 저장
if circles is not None:
    circles = np.round(circles[0, :]).astype(int)
    for (x, y, r) in circles:
        df = df.append({'x': x, 'y': y}, ignore_index=True)

# 결과 출력
print(df)

# 좌표 표시
for _, row in df.iterrows():
    x = int(row['x'])
    y = int(row['y'])
    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # 빨간색 원으로 좌표 표시

# 이미지 출력
cv2.imshow('Image with Coordinates', image)
cv2.waitKey(0)
cv2.destroyAllWindows()