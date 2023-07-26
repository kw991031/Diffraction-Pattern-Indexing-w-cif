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
    else:
        scale_bar_length = "N/A"
    
    # 입력값 출력
    print("Scale Bar Length:", scale_bar_length)
    return w, scale_bar_length
def crop_image(selected_filename):
    # 이미지 파일 경로
    image_path = selected_filename

    # 이미지 로드
    image = cv2.imread(image_path)

    # 이미지 표시 윈도우 생성
    cv2.namedWindow("이미지 선택")
    cv2.imshow("이미지 선택", image)

    # 마우스 이벤트 처리
    start_x, start_y = 0, 0
    end_x, end_y = 0, 0
    dragging = False

    def on_mouse_event(event, x, y, flags, param):
        global dragging, start_x, start_y, end_x, end_y

        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            start_x, start_y = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
            end_x, end_y = x, y
            # 선택한 범위 출력
            print("선택한 범위: x={0}, y={1}, width={2}, height={3}".format(start_x, start_y, end_x-start_x, end_y-start_y))

        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging:
                # 드래그 중인 경우 사각형 그리기
                current_x, current_y = x, y
                cv2.rectangle(image, (start_x, start_y), (current_x, current_y), (0, 255, 0), 2)
                cv2.imshow("이미지 선택", image)

    # 마우스 이벤트 콜백 등록
    cv2.setMouseCallback("이미지 선택", on_mouse_event)

    # 사용자 입력 대기
    cv2.waitKey(0)

    # 윈도우 종료 및 리소스 정리
    cv2.destroyAllWindows()

    # 이미지 로드
    image = cv2.imread(selected_filename)

    # 선택한 범위만을 포함하는 이미지 추출
    cropped_image = image[start_y:end_y, start_x:end_x]

    # 새로운 이미지 표시
    cv2.imshow("Cropped Image", cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Tkinter 창 생성
root = tk.Tk()
root.withdraw()

# 파일 브라우저 열기
file_path = filedialog.askopenfilename()

# 파일명 추출
selected_filename = file_path.split('/')[-1]

print("Selected File:", selected_filename)


crop_image(selected_filename)


width, scale_bar_length=get_scalebar(selected_filename)

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

# 이미지 위에 좌표 표시
for index, row in df.iterrows():
    x = int(row['X'])
    y = int(row['Y'])
    cv2.putText(image, f"{index}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# 이미지 출력
cv2.imshow('Image with Coords', image)
cv2.waitKey(0)
cv2.destroyAllWindows()