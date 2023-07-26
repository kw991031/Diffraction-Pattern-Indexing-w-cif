import cv2
import numpy as np
import math
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox, simpledialog
import os

# 마우스 이벤트 처리
start_x, start_y = 0, 0
end_x, end_y = 0, 0
dragging = False
cropping = False
#Scale bar 인식하고 길이를 입력 받는 함수
def get_scalebar(selected_filename):

        # 선택지 버튼 클릭 시 실행되는 함수
    def handle_button_click(choice):
        global answer
        answer = choice
        window.destroy()

    # 이미지 파일 읽기
    image = cv2.imread(selected_filename)

    # 이미지 전처리
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # 윤곽선 검출
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 사각형 찾기
    rectangles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        rectangles.append((x, y, w, h))

    # 긴 사각형 찾기
    longest_rectangle = max(rectangles, key=lambda r: r[2])

    while True:
        # 이미지 파일 읽기
        image = cv2.imread(selected_filename)

        # 이미지에 사각형 표시
        x, y, w, h = longest_rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 이미지 출력
        cv2.imshow("Longest Rectangle", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 사용자 확인을 위한 창 생성
        root = tk.Tk()
        root.withdraw()

        # 사용자 정의 창 생성
        window = tk.Toplevel(root)
        window.title("Confirmation")

        # 선택지 버튼 생성
        yes_button = tk.Button(window, text="Yes", command=lambda: handle_button_click("Yes"))
        no_button = tk.Button(window, text="No", command=lambda: handle_button_click("No"))
        skip_button = tk.Button(window, text="No rectangle here", command=lambda: handle_button_click("No rectangle here"))

        # 선택지 버튼 배치
        yes_button.pack(padx=10, pady=5)
        no_button.pack(padx=10, pady=5)
        skip_button.pack(padx=10, pady=5)

        # 사용자 확인 창 표시
        window.wait_window(window)

        if answer == "Yes":
            # 이미지에 사각형 표시
            x, y, w, h = longest_rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 이미지 출력
            cv2.imshow("Longest Rectangle", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # 사용자가 "Yes" 선택한 경우, 다음 사각형으로 이동
            break
        elif answer == "No rectangle here":
            # 사용자가 "No rectangle here" 선택한 경우, 건너뛰고 반복 종료
            break
        else:
            # 사용자가 "No" 선택한 경우, 다음으로 긴 사각형 찾기
            rectangles.remove(longest_rectangle)
            longest_rectangle = max(rectangles, key=lambda r: r[2] * r[3])
    
    if answer != "No rectangle here":
        # 사용자에게 입력 받기
        root = tk.Tk()
        root.withdraw()

        scale_bar_length = simpledialog.askfloat("Input", "몇 1/nm 입니까?")
        per_pixel = scale_bar_length/w# 입력값 출력
        print("nm-1 per pixel:", per_pixel)
        return per_pixel
    else:
        scale_bar_length = "N/A"
        return "N/A"
    
    

# 전역 변수 선언
x_start, y_start, x_end, y_end = 0, 0, 0, 0
cropping = False
crop = None

# 마우스 이벤트 처리 함수
def crop_image(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, cropping, image_copy, crop
    
    # 이미지 복사
    image_copy = image.copy()
    
    # 마우스 왼쪽 버튼을 누를 때
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start = x, y
        x_end, y_end = x, y
        cropping = True
    
    # 마우스 이동 시
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y
            # 시작 좌표와 종료 좌표로 사각형 그리기
            cv2.rectangle(image_copy, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.imshow("Original Image", image_copy)
    
    # 마우스 왼쪽 버튼을 놓을 때
    elif event == cv2.EVENT_LBUTTONUP:
        x_end, y_end = x, y
        cropping = False
        # crop 함수 호출
        crop_window()
        
# crop 함수
def crop_window():
    global x_start, y_start, x_end, y_end, image, image_copy, crop
    
    # 시작 좌표와 종료 좌표로 사각형 그리기
    cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    cv2.imshow("Original Image", image)
    
    # crop 영역 추출
    crop = image_copy[y_start:y_end, x_start:x_end]
    
    # crop된 이미지 새로운 창에 표시
    cv2.imshow("Cropped Image", crop)

    
# Tkinter 창 생성
root = tk.Tk()
root.withdraw()

# 파일 브라우저 열기
file_path = filedialog.askopenfilename()

# 파일명 추출
selected_filename = file_path.split('/')[-1]

print("Selected File:", selected_filename)

per_pixel=get_scalebar(selected_filename)


#Scale bar가 존재할 때 제외한 영역을 선택해 image에 저장
if per_pixel != "N/A":
    # 이미지 로드
    image = cv2.imread(selected_filename)

    # 이미지 복사
    image_copy = image.copy()

    # 이미지 윈도우 생성 및 이벤트 콜백 함수 등록
    cv2.namedWindow("Select area without scale bar")
    cv2.setMouseCallback("Select area without scale bar", crop_image)

    # 이미지 출력
    cv2.imshow("Select area without scale bar", image)

    # 키 입력 대기
    cv2.waitKey(0)

    # crop 영역이 존재하는 경우에만 crop 이미지 출력
    if crop is not None:
        image=crop
        cv2.imshow("Cropped Image", image)

    # 키 입력 대기
    cv2.waitKey(0)

    # 윈도우 종료
    cv2.destroyAllWindows()

#이미지 처리 시작----------------------------------------------------------------------------------------------

# 이미지 파일 읽기
image = cv2.imread(selected_filename, 0)
if per_pixel is not "N/A":
    image = image[y_start:y_end, x_start:x_end]


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
df['d'] = 10/(df['R'] * per_pixel)

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