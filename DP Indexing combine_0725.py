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

#이미지 처리 시작----------------------------------------------------------------------------------------------

# 이미지 파일 읽기
image = cv2.imread(selected_filename, 0)
if per_pixel != "N/A":
    image = image[y_start:y_end, x_start:x_end]


# 이미지 전처리
ret, threshold = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)  # 임계값 적용하여 흑백 이진화

cv2.imshow('any',threshold)
# 윤곽선 검출
contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 이미지 중심점 계산
center_x = image.shape[1] / 2  # 정수 나눗셈 대신 실수 나눗셈 사용
center_y = image.shape[0] / 2

# 가장 큰 컨투어 찾기
largest_contour = max(contours, key=cv2.contourArea)

# 가장 큰 컨투어의 중심 좌표 계산
M = cv2.moments(largest_contour)
largest_center_x = M["m10"] / M["m00"] if M["m00"] != 0 else 0  # 정수 나눗셈 대신 실수 나눗셈 사용
largest_center_y = M["m01"] / M["m00"] if M["m00"] != 0 else 0  # 정수 나눗셈 대신 실수 나눗셈 사용


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
    center_x = M["m10"] / M["m00"] if M["m00"] != 0 else 0  # 정수 나눗셈 대신 실수 나눗셈 사용
    center_y = M["m01"] / M["m00"] if M["m00"] != 0 else 0  # 정수 나눗셈 대신 실수 나눗셈 사용

    # 극좌표 계산
    dx = center_x - largest_center_x
    dy = center_y - largest_center_y
    radius = math.sqrt(dx ** 2 + dy ** 2)
    theta = math.atan2(dy, dx)

    # 각도를 0~360 범위로 변환
    if theta < 0:
        theta += 2 * math.pi

    data.append({'X': center_x, 'Y': center_y, 'R': radius, 'Theta': math.degrees(theta)})

df = pd.DataFrame(data)

# DataFrame에서 R이 두 번째, 세 번째로 작은 값을 선택
df_new = df[df['Theta'] <= 180].copy()
sorted_df = df_new.sort_values('R')

print(sorted_df)


# 두 번째 행의 정보 가져오기
second_row = sorted_df.iloc[1]
second_row_theta = second_row['Theta']

# R 값과 Theta 값이 비슷한 행 찾기
error_threshold = 0.1
similar_rows = df[
    (abs(df['R'] - second_row['R']) / second_row['R'] <= error_threshold) &
    (abs(df['Theta'] - (second_row_theta + 180))/180 <= error_threshold)
]

# 선택된 행과 두 번째 행으로 이루어진 DataFrame 생성
first_df = pd.concat([pd.DataFrame([second_row]), similar_rows])

#세 번째 행에 대해 똑같은 것을 반복하고 selected_df에 추가
third_row=sorted_df.iloc[2]
third_row_theta=third_row['Theta']

similar_rows=df[
    (abs(df['R'] - third_row['R']) / third_row['R'] <= error_threshold) &
    (abs(df['Theta'] - (third_row_theta+180))/180 <= error_threshold)
]
second_df = pd.concat([pd.DataFrame([third_row]), similar_rows])

# 결과 출력
print("first_df:\n",first_df)
print("second_df:\n",second_df)

first_center= {'X':(first_df.iloc[0]['X']+first_df.iloc[1]['X'])/2, 'Y':(first_df.iloc[0]['Y']+first_df.iloc[1]['Y'])/2}
print(first_center)
second_center={'X':(second_df.iloc[0]['X']+second_df.iloc[1]['X'])/2, 'Y':(second_df.iloc[0]['Y']+second_df.iloc[1]['Y'])/2}
print(second_center)

final_center={'X': (first_center['X']+second_center['X']+sorted_df.iloc[0]['X'])/3, 'Y': (first_center['Y']+second_center['Y']+sorted_df.iloc[0]['Y'])/3}
print("final center:\n", final_center)

# R과 Theta 열 초기화
df['R'] = 0.0
df['Theta'] = 0.0

# R과 Theta 열 계산
for index, row in df.iterrows():
    dx = row['X'] - final_center['X']
    dy = row['Y'] - final_center['Y']
    df.at[index, 'R'] = math.sqrt(dx ** 2 + dy ** 2)
    theta = math.atan2(dy, dx)
    if theta < 0:
        theta += 2 * math.pi

    df.at[index, 'Theta'] = math.degrees(theta)

df=df.sort_values('R')

if per_pixel=='N/A': per_pixel=0
df['d(Å)'] = 10/(df['R'] * per_pixel)
print(df)

# DataFrame에서 R이 두 번째, 세 번째로 작은 값을 선택
sorted_df = df[df['Theta'] <= 180].copy()
sorted_df = sorted_df.sort_values('R')
sorted_df.iloc[0, :] = [final_center['X'], final_center['Y'],0,0,0]

#basis vector 구하기
r0 = sorted_df.iloc[0]
r1 = sorted_df.iloc[1]
r2 = sorted_df.iloc[2]

g1 = r1 - r0
g2 = r2 - r0

basis_df = pd.DataFrame({'g1': g1, 'g2': g2}).transpose()
print(basis_df)

#CIF 파일 처리-----------------------------------------------------------------------------------------------------------------------------------------

folder_path = 'C:/Users/KIST/Desktop/CIF'  # 작업할 폴더의 경로 설정
dataframes = []  # DataFrame을 저장할 리스트 생성
raw_dataframes=[]
g_groups=[]
err_dict={}
lat_par=[]

# # 데이터 딕셔너리 생성
# data = {
#     'X': [basis_df['X'].iloc[0], basis_df['X'].iloc[1]],
#     'Y': [basis_df['Y'].iloc[0], basis_df['Y'].iloc[1]],
#     'R': [basis_df['R'].iloc[0], basis_df['R'].iloc[1]],
#     'Theta': [basis_df['Theta'].iloc[0], basis_df['Theta'].iloc[1]],
#     'd(Å)': [basis_df['d(Å)'].iloc[0], basis_df['d(Å)'].iloc[1]]
# }

# # 데이터프레임 생성
# basis_df = pd.DataFrame(data, index=['g1', 'g2'])

def trim_cif(name):
    #lattice parameter 구하기
    # h가 1, k가 0, l이 0인 행을 찾아서 'd(Å)' 값을 변수 'a'에 저장
    condition = (name['h'] == 1) & (name['k'] == 0) & (name['l'] == 0)
    a = name.loc[condition, 'd(Å)'].values[0]
    condition = (name['h'] == 0) & (name['k'] == 1) & (name['l'] == 0)
    b = name.loc[condition, 'd(Å)'].values[0]
    condition = (name['h'] == 0) & (name['k'] == 0) & (name['l'] == 1)
    c = name.loc[condition, 'd(Å)'].values[0]
    print('lattice parameter: ', a, b, c)

    # 'I' 열이 0인 값을 제외
    name = name[name['I'] != 0]

    # 'd(Å)' 열의 크기 순으로 내림차순 정렬
    name = name.copy()  # 'name' 데이터프레임을 사본으로 만듦
    name['d(Å)'] = name['d(Å)'].astype(float)
    name = name.sort_values('d(Å)', ascending=False)

    # 'd(Å)' 열을 기준으로 groupby한 후 재인덱싱
    grouped_df = name.groupby('d(Å)', as_index=False).apply(lambda x: x.reset_index(drop=True))

    # 'd(Å)' 열의 큰 값부터 1씩 감소시키며 그룹 번호를 부여
    max_d = grouped_df['d(Å)'].max()
    grouped_df['Group'] = (max_d - grouped_df['d(Å)'] + 1).rank(method='dense').astype(int)

    # 그룹 번호를 오름차순으로 정렬
    grouped_df = grouped_df.sort_values('Group', ascending=True)

    return grouped_df, a,b,c

def get_angle(phase,group1,i1,group2,i2):
    a = lat_par[lat_par['name'] == phase]['a'].values[0]
    b = lat_par[lat_par['name'] == phase]['b'].values[0]
    c = lat_par[lat_par['name'] == phase]['c'].values[0]
    alpha=90*(math.pi/180)
    beta=90*(math.pi/180)
    gamma=90*(math.pi/180)

    V=a*b*c*math.sqrt(1-(math.cos(alpha))**2-(math.cos(beta))**2-(math.cos(gamma))**2+2*math.cos(alpha)*math.cos(beta)*math.cos(gamma))

    def get_angle_star(x,y,z):
        cosx_star=(math.cos(y)*math.cos(z)-math.cos(x))/(math.sin(y)*math.sin(z))
        x_star=math.acos(cosx_star)
        return x_star
    alpha_star=get_angle_star(alpha,beta,gamma)
    beta_star=get_angle_star(beta,alpha,gamma)
    gamma_star=get_angle_star(gamma,alpha,beta)

    def get_star(b,c,alpha):
        a_star=b*c*math.sin(alpha)/V
        return a_star
    a_star=get_star(b,c,alpha)
    b_star=get_star(c,a,beta)
    c_star=get_star(a,b,gamma)
    
    row=group1.iloc[i1]
    d1=row['d(Å)']
    h1=row['h']
    k1=row['k']
    l1=row['l']
    
    row=group2.iloc[i2]
    d2=row['d(Å)']
    h2=row['h']
    k2=row['k']
    l2=row['l']

    cosangle=d1*d2*(h1*h2*a_star**2+k1*k2*b_star**2+l1*l2*c_star**2+(k1*l2+l1*k2)*b_star*c_star*math.cos(alpha_star)+(h1*l2+l1*h2)*a_star*c_star*math.cos(beta_star)+(h1*k2+k1*h2)*a_star*b_star*math.cos(gamma_star))
    if 1 <= abs(cosangle) < 1.05:
        cosangle = 1
    angle_rad=math.acos(cosangle)
    angle=angle_rad*180/math.pi
    return angle

print (basis_df)
# 폴더 내의 모든 파일을 확인하여 dataframes에 저장
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):  # txt 파일인 경우에만 작업 수행
        file_path = os.path.join(folder_path, filename)  # 파일 경로 생성

        # 파일 읽기
        df = pd.read_csv(file_path, delimiter='\s+', skiprows=[0], names=['h', 'k', 'l', 'd(Å)', 'F(real)', 'F(imag)', '|F|', '2θ', 'I', 'M', 'ID(λ)', 'Phase'])
        df = df.sort_values('d(Å)', ascending=False)       
        df,a,b,c = trim_cif(df)

        # 파일명에서 확장자를 제외한 부분 추출
        file_name = os.path.splitext(filename)[0]

        # DataFrame에 파일명 할당
        df.name = file_name
        
        data=[file_name,a,b,c] #trim 전 데이터로부터 100 010 001 plane distance 사용해 lattice parameter 인식
        lat_par.append(data)

        # DataFrame을 리스트에 추가
        dataframes.append(df)

lat_par = pd.DataFrame(lat_par, columns=['name', 'a', 'b', 'c'])

print(lat_par)

# 각각의 cif파일에서 작업


for idx, df in enumerate(dataframes): #cif파일 하나씩 처리하고 반복 (기저벡터와 유사한 plane spacing 찾기)
    g_groups = []  # 오차율이 가장 작은 그룹들을 저장할 리스트
    err_sum=0
    min_ang_err= float('inf')
    min_error = float('inf')
    selected_group = None
    selected_group_name = None  # 선택된 그룹의 이름을 저장할 변수 추가
    print(f"DataFrame {idx+1}: {df.name}")
    
    for index, row in basis_df.iterrows(): #주어진 basis dataframe 에 대해 반복하여 기저벡터 길이와 유사한 plane spacing group을 찾기
        given_d = row['d(Å)']
        d_values = df['d(Å)'].values
        errors = abs((d_values - given_d) / given_d)
        min_error_idx = errors.argmin()
        if errors[min_error_idx] < min_error:
            min_error = errors[min_error_idx]
            selected_group = df.iloc[[min_error_idx]]
            
        
        #phase 간의 정확성 비교 위해 오차의 총합 구하기
        err_sum = err_sum + min_error

        # 선택된 그룹과 같은 group 값을 갖는 행들을 selected_group에 저장
        selected_group_temp = df[df['Group'] == selected_group['Group'].iloc[0]]

        # 선택된 그룹을 새로운 DataFrame으로 만들어 저장 (이름과 함께)
        selected_group_name = f"{index}_group - ({df.name})"
        selected_group = pd.DataFrame(selected_group_temp)
        selected_group.name = selected_group_name
        g_groups.append(selected_group)

        # 선택된 그룹의 데이터 출력
        print("Selected Group Name:", selected_group_name)
        print("Selected Group:")
        print(selected_group)
        print('\n')
    
    # 두 개 그룹간의 반복을 통해 theta와 유사한 각도가 있는지 확인하여 오차를 저장
    g1 = g_groups[0]
    g2 = g_groups[1]
    basis_ang = abs(basis_df['Theta'].iloc[0] - basis_df['Theta'].iloc[1])  # basis vector 간의 차를 이용해 각도를 구함
    
    # 각 데이터프레임의 이름과 행 수 구하기
    n_rows_group1 = len(g1)
    n_rows_group2 = len(g2)

    # 두 데이터프레임의 모든 행(row) 간의 각도 계산
    for i1 in range(n_rows_group1):
        for i2 in range(n_rows_group2):
            angle = get_angle(df.name, g_groups[0], i1, g_groups[1], i2)  # get angle 함수 사용해서 두 그룹 내 하나씩 반복해서 각도를 구함
            ang_err = abs(basis_ang - angle) / angle  # 오차 반영을 위해 각도 오차를 계산
            if ang_err < min_ang_err:
                min_ang_err = ang_err
                min_g1 = g_groups[0].iloc[i1]
                min_g2 = g_groups[1].iloc[i2]
    print(df.name,'의 각도 에러값입니다: ',min_ang_err,'\n')

    err_sum += min_ang_err
    name=df.name
    err=err_sum
    err_dict[name]=err

# DataFrame에 저장하기
err_df = pd.DataFrame(err_dict.items(), columns=['Name', 'Error'])

# 결과 출력
print(err_df)