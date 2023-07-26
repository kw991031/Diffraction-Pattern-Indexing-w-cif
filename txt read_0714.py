import pandas as pd
import tkinter as tk
from tkinter import filedialog

# Tkinter 창 생성
root = tk.Tk()
root.withdraw()

# 파일 브라우저 열기
file_path = filedialog.askopenfilename()

# 파일 읽기
df = pd.read_csv(file_path, delimiter='\s+', skiprows=[0], names=['h', 'k', 'l', 'd(Å)', 'F(real)', 'F(imag)', '|F|', '2θ', 'I', 'M', 'ID(λ)', 'Phase'])

# 'I' 열이 0인 값을 제외
df = df[df['I'] != 0]

# 'd(Å)' 열의 크기 순으로 내림차순 정렬
df['d(Å)'] = df['d(Å)'].astype(float)
df = df.sort_values('d(Å)', ascending=False)

grouped_df = df.groupby('d(Å)').apply(lambda x: x.reset_index(drop=True))
# grouped_df.reset_index(drop=True, level=0, inplace=True)
# grouped_df.index = pd.Series(range(1, len(grouped_df) + 1))

# 결과 출력
print(grouped_df)

# 결과 저장
output_file_path = 'output.txt'
df.to_csv(output_file_path, index=False, sep='\t')
print(f"결과가 {output_file_path} 파일로 저장되었습니다.")
