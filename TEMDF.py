import cv2
import numpy as np
import math

# 이미지 불러오기
image = cv2.imread('tem_image2.jpg', 0)  # 흑백 이미지로 로드

# 이미지 전처리
ret, threshold = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)  # 임계값 적용하여 흑백 이진화

# 윤곽선 검출
contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 극좌표 계산 및 표시
for contour in contours:
    # 면적이 작은 윤곽선 제외
    area = cv2.contourArea(contour)
    if area < 1:
        continue

    # 윤곽선 중심 좌표 계산
    M = cv2.moments(contour)
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])

    # 극좌표 계산
    dx = center_x - image.shape[1] // 2
    dy = center_y - image.shape[0] // 2
    radius = math.sqrt(dx ** 2 + dy ** 2)
    theta = math.atan2(dy, dx)

    # 각도를 0~360 범위로 변환
    if theta < 0:
        theta += 2 * math.pi

    # 극좌표 표시
    cv2.circle(image, (center_x, center_y), 5, (0, 255, 0), -1)
    cv2.putText(image, f"R:{radius:.2f} T:{math.degrees(theta):.2f}", (center_x - 50, center_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# 결과 이미지 출력
cv2.imshow('Polar Coordinates', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
