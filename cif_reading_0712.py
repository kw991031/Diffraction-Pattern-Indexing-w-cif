import pandas as pd
import tkinter as tk
from tkinter import filedialog

# Tkinter를 사용하여 파일 선택 대화상자 열기
root = tk.Tk()
root.withdraw()

# 파일 선택
file_path = filedialog.askopenfilename()

# 파일 열기 및 내용 수정
with open(file_path, 'r') as file:
    file_contents = file.read()
    modified_contents = file_contents.replace("d (Å)", "d(Å)")

# 수정된 내용을 임시 파일에 저장
temp_file_path = 'temp.txt'
with open(temp_file_path, 'w') as temp_file:
    temp_file.write(modified_contents)

# 임시 파일을 DataFrame으로 읽어오기
df = pd.read_csv(temp_file_path, sep='\s+')

# DataFrame 출력
print(df)

# 임시 파일 삭제
import os
os.remove(temp_file_path)

filtered_df = df[df['I'] >= 0.01]

# 결과 출력
print(filtered_df)