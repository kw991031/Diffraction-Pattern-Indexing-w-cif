import os
import pandas as pd
import math

folder_path = 'C:/Users/KIST/Desktop/CIF'  # 작업할 폴더의 경로 설정
dataframes = []  # DataFrame을 저장할 리스트 생성
raw_dataframes=[]
g_groups=[]
err_dict={}
lat_par=[]

# 데이터 딕셔너리 생성
data = {
    'X': [33.764545, -4.328397],
    'Y': [2.481679, 55.624517],
    'R': [33.855623, 55.792668],
    'Theta': [4.203656, 94.449479],
    'd': [4.78, 2.87]
}

# 데이터프레임 생성
basis_df = pd.DataFrame(data, index=['g1', 'g2'])

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

# 폴더 내의 모든 파일을 확인하여 dataframes에 저장
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):  # txt 파일인 경우에만 작업 수행
        file_path = os.path.join(folder_path, filename)  # 파일 경로 생성

        # 파일 읽기
        df = pd.read_csv(file_path, delimiter='\s+', skiprows=[0], names=['h', 'k', 'l', 'd(Å)', 'F(real)', 'F(imag)', '|F|', '2θ', 'I', 'M', 'ID(λ)', 'Phase'])
        df = df.sort_values('d(Å)', ascending=False)
        raw_df=df
        df,a,b,c = trim_cif(df)

        # 파일명에서 확장자를 제외한 부분 추출
        file_name = os.path.splitext(filename)[0]

        # DataFrame에 파일명 할당
        df.name = file_name
        raw_df.name = file_name

        data=[file_name,a,b,c]
        lat_par.append(data)

        # DataFrame을 리스트에 추가
        dataframes.append(df)
        raw_dataframes.append(raw_df)

lat_par = pd.DataFrame(lat_par, columns=['name', 'a', 'b', 'c'])

print(lat_par)

# 각각의 cif파일에서 작업
g_groups = []  # 오차율이 가장 작은 그룹들을 저장할 리스트

for idx, df in enumerate(dataframes):
    err_sum=0
    min_error = float('inf')
    selected_group = None
    selected_group_name = None  # 선택된 그룹의 이름을 저장할 변수 추가
    print(f"DataFrame {idx+1}: {df.name}")
    
    for index, row in basis_df.iterrows():
        given_d = row['d']
        d_values = df['d(Å)'].values
        errors = abs((d_values - given_d) / given_d)
        min_error_idx = errors.argmin()
        if errors[min_error_idx] < min_error:
            min_error = errors[min_error_idx]
            selected_group = df.iloc[min_error_idx]

            selected_group_name = f"{index}_group"  # 선택된 그룹의 이름 저장
        
        #phase 간의 정확성 비교 위해 오차의 총합 구하기
        err_sum = err_sum + min_error

        # 선택된 그룹과 같은 group 값을 갖는 행들을 selected_group에 저장
        selected_group = df[df['Group'] == selected_group['Group']]
        selected_group.name=df.name + ":" + selected_group_name
        

        # 선택된 그룹을 리스트에 추가 (이름과 함께)
        g_groups.append(selected_group)

        # 선택된 그룹의 데이터 출력
        print("Selected Group Name:", selected_group_name)
        print("Selected Group:")
        print(selected_group)
        print('\n')
    name=df.name
    err=err_sum
    err_dict[name]=err
    
print(err_dict)

print(g_groups)

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

    d1=group1['d(Å)'].iloc[i1]
    h1=group1['h'].iloc[i1]
    k1=group1['k'].iloc[i1]
    l1=group1['l'].iloc[i1]
    

    d2=group2['d(Å)'].iloc[i2]
    h2=group2['h'].iloc[i2]
    k2=group2['k'].iloc[i2]
    l2=group2['l'].iloc[i2]

    cosangle=d1*d2*(h1*h2*a_star**2+k1*k2*b_star**2+l1*l2*c_star**2+(k1*l2+l1*k2)*b_star*c_star*math.cos(alpha_star)+(h1*l2+l1*h2)*a_star*c_star*math.cos(beta_star)+(h1*k2+k1*h2)*a_star*b_star*math.cos(gamma_star))
    angle_rad=math.acos(cosangle)
    angle=angle_rad*180/math.pi
    return cosangle, angle

cosangle,angle=get_angle('PDF Card tetragonal - 04-014-3126',g_groups[2],0,g_groups[3],2)
print(cosangle)
print(angle)
print(g_groups[3].name)


