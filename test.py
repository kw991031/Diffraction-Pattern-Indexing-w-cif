import cv2
import numpy as np
import math
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox, simpledialog
import os
from scipy.optimize import least_squares

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
            cv2.imshow("Select area without scale bar", image_copy)
    
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
    # 키 입력 대기
    cv2.waitKey(0)

    # 윈도우 종료
    cv2.destroyAllWindows()

    
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
    # cv2.namedWindow("Select area without scale bar")
    # 이미지 출력
    cv2.imshow("Select area without scale bar", image)
    cv2.setMouseCallback("Select area without scale bar", crop_image)



    # 키 입력 대기
    cv2.waitKey(0)

    # crop 영역이 존재하는 경우에만 crop 이미지 출력
    # if crop is not None:
    #     image=crop
    #     cv2.imshow("Cropped Image", image)

    # 키 입력 대기
    cv2.waitKey(0)

    # 윈도우 종료
    cv2.destroyAllWindows()
else:  
    image=cv2.imread(selected_filename)


def circle_residuals(params, x, y):
    # params: [x_center, y_center, radius]
    residuals = np.sqrt((x - params[0]) ** 2 + (y - params[1]) ** 2) - params[2]
    return residuals

def fit_circle(x, y):
    # Initial estimation for circle parameters
    x_center_initial = np.mean(x)
    y_center_initial = np.mean(y)
    radius_initial = np.mean(np.sqrt((x - x_center_initial) ** 2 + (y - y_center_initial) ** 2))

    # Perform least squares circle fitting
    params_initial = [x_center_initial, y_center_initial, radius_initial]
    result = least_squares(circle_residuals, params_initial, args=(x, y))

    # Extract the fitted circle parameters
    x_center = result.x[0]
    y_center = result.x[1]
    radius = result.x[2]

    return x_center, y_center, radius

# 이미지 파일 읽기
image = cv2.imread(selected_filename, cv2.IMREAD_GRAYSCALE)  # 그레이스케일로 이미지 읽기
if per_pixel != "N/A":
    image = image[y_start:y_end, x_start:x_end]

# 이미지 전처리
ret, threshold = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)  # 임계값 적용하여 흑백 이진화

# 윤곽선 검출
contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 이미지에 컨투어 그리기
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# 극좌표 계산 및 DataFrame 생성
data = []
for contour in contours:
    # 윤곽선 중심 좌표 계산
    M = cv2.moments(contour)
    center_x = M["m10"] / M["m00"] if M["m00"] != 0 else 0
    center_y = M["m01"] / M["m00"] if M["m00"] != 0 else 0

    # Fit circle to the contour points
    x = contour[:, 0, 0]
    y = contour[:, 0, 1]
    x_center, y_center, radius = fit_circle(x, y)

    data.append({'X': x_center, 'Y': y_center, 'R': radius})

# DataFrame 생성
df = pd.DataFrame(data)
df = df.sort_values('R')
print(df)