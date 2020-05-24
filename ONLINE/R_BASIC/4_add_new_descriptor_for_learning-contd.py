
# coding: utf-8

# # 네번째. SDR Rest API 을 이용한 학습 데이터 편집: 새로운 Descriptor 추가하기
# <hr/>
# ## 예제 목표:
#   이번 예제에서는 SDR Rest API를 활용하여 기축된 소재 Database에서 사용자가 원하는 종류의 데이터를 추출하고, 추가 테이블 값을 참조하여 새로운 descriptor 를 생성하며, 마지막으로 새로 만들어진 학습 데이터를 활용하여 Deep Learning을 수행하려고 한다. 데이터 추출은 lucene query 를 기반으로 만들어진 SDR RESTful API 를 이용한다. 이해를 돕기 위해 먼저 복잡한 형태의 query 를 먼저 수행하여 작은 양의 데이터를 추출하여 average electronegativity 등의 물성 값 계산을 수행하고 학습 데이터를 building한다. 이후 많은 양의 데이터를 추출한 후 가시화 기법등을 사용하여 데이터를 검증 및 분석한다.이전의 예제들과 마찬가지로, Materials Scientific Community에 기축되어 있는 Open Quantum Materials Database [[1]](http://dx.doi.org/10.1007/s11837-013-0755-4)[[2]](http://dx.doi.org/10.1038/npjcompumats.2015.10) ([OQMD](http://www.oqmd.org), NorthWestern Univ.)의 DFT 계산 데이터를 기반으로 예제를 학습해본다.
#   
# ### Overview
#   예제는 크게 다음의 순서로 구성된다.
# 1. 데이터 추출 (Data Extraction)
# <br/> 1.1. 작은 양의 데이터 추출 (by Searching)
# <br><br>
# 2. 학습 데이터 빌드 (Building Learning Datasets)
# <br/> 2.1. 화학론량적 계산값 추가 - Electronegativity
# <br/> 2.2. 데이터 프레임 병합(Merging)
# <br><br>
# 3. 심화된 학습 데이터 분석 및 가시화
# <br/> 3.1. Binary Compounds: 데이터 추출
# <br/> 3.2. Binary Compounds: 학습 데이터 빌드
# <br/> 3.3. Binary Compounds: 데이터 가시화
# <br/> 3.4. 전체 데이터 추출 (by Crawling)
# <hr/>

# ### 1. 데이터 추출 (Data Extraction)
# #### 1.1. 작은 양의 데이터 추출 (by Searching)
# 데이터를 추출하기에 앞서 SDR REST API 의 동작과정을 간단히 설명한다. 먼저, 유저가 소재 웹 페이지의 Advanced Search Page 와 동일한 형태의 lucene based syntax 를 이용하여 https 패킷을 생성하여 서버에 전달한다. 해당 패킷은 사용자 id와 password 등을 포함하여 암호화되어 서버에 전달된다. 서버는 유저의 인증 과정을 거친 후 쿼리 부분을 추출하여 데이터베이스를 검색한 후, 매칭이 되는 데이터를 유저에게 반환한다. 서버로 부터 전달받은 데이터는 유저의 개발 환경에 json 포멧의 파일로 저장된다.

# In[1]:


########################################
## path settings #######################
########################################
#
# path setting
currentPath = "./"
dataPath = "data/"
figPath = ".figures/"
modelPath = "models/"
data_name = "final_energy_per_atom_"


# 보안 상 user_id, user_pwd, sever_address 부분은 임의로 작성해 두었다. 실습을 수행하려면 해당 부분을 적절한 내용으로 대체 후 실행해야한다. 예제에서는 쿼리 부분의 공백과 소수점 입력등을 원활히 하기위해 curl 대신 wget 명령어를 사용하였다. 이번 예제에서는 oqmd 타입의 데이터들 중 Li를 포함하면서 원소 종류가 6개인 화합물을 검색하였다.
# ** 개인인증서 lets encrypt 를 이후에 더 설명해아할지 고민.

# In[2]:


import subprocess
user_id = "NONE"
user_pwd = "NONE"

extra_args = "--no-check-certificate"
server_address = "NONE"
rest_api_option = "/rest/api/search/"

#데이터 타입 설정
data_type = "oqmd"
basic_lucene_query = "DataType:"+data_type+" AND "

#쿼리 설정
other_lucene_query = "elements: Li AND nelements:6"

full_query = '"'+ server_address + rest_api_option + basic_lucene_query + other_lucene_query + '"'
jsonResultFileName = "query_result.json"

command_line = 'wget -O' + ' ' + currentPath + dataPath + jsonResultFileName +' '+'--user'+ ' '+ user_id + ' '+ '--password' + ' ' + user_pwd + ' ' + extra_args + ' ' + full_query

subprocess.call(command_line, shell=True)


# In[3]:


#만들어진 명령어 확인
command_line


# Datatype, collectionId, datasetId 등의 정보를 포함하는 datasets의 리스트가 json 형태로 반환됨을 확인할 수 있다.

# In[4]:


import pandas as pd
query_result = pd.read_json(currentPath+dataPath+jsonResultFileName)
query_result


# 총 4개의 dataset이 확인되었다. 이 중 하나의 dataset의 상세정보를 확인해본다. unitcelformula, finalenergy, density, volume, lattice 등의 소재 정보와 collectionId, datasetId, userId 등의 관리 데이터를 확인할 수 있다.

# In[5]:


query_result.loc[0]


# ### 2. 학습 데이터 빌드 (Building Learning Datasets)
# #### 2.1. 화학론량적 계산값 추가 - Electronegativity

# \\(  T^{avg}_{A_xB_yC_z} = \frac{xT_{A}}{x+y+z} + \frac{yT_{B}}{x+y+z} + \frac{zT_{C}}{x+y+z} \\) 식을 사용하여, \\(Cs_2Li_2H_8S_4N_4O_{12} \\)(query_result.loc[0])의 Electronegativity 를 계산해본다. 이를 위해 각 원소 별 electronegativity 값을 참조 테이블에서 확인해보자.

# In[6]:


fileName = "reference_elements_dataset.csv"
atomtable = pd.read_csv(currentPath+dataPath+fileName)


# In[7]:


atomtable.head()


# 각 원소별 electronegativity 값은 다음과 같이 확인할 수 있다.

# In[8]:


print('Cs', atomtable.loc[atomtable['symbol']=='Cs'].electronegativity.values)
print('Li', atomtable.loc[atomtable['symbol']=='Li'].electronegativity.values)
print('H', atomtable.loc[atomtable['symbol']=='H'].electronegativity.values)
print('S', atomtable.loc[atomtable['symbol']=='S'].electronegativity.values)
print('N', atomtable.loc[atomtable['symbol']=='N'].electronegativity.values)
print('O', atomtable.loc[atomtable['symbol']=='O'].electronegativity.values)


# dict 형태의 unitcellformula 정보로부터 avg_electronegativity 를 계산하는 함수를 간단히 표현하면 다음과 같다.

# In[9]:


def getElectronegativity_from_dict(dict_comp, atomtable):
    total_num_of_atoms = 0
    sum_of_electronegativity = 0
    for key, value in dict_comp.items():
        sum_of_electronegativity += value * float(atomtable.loc[atomtable['symbol']==key].electronegativity.values)
        total_num_of_atoms += value
    return sum_of_electronegativity/total_num_of_atoms


# 이제 query_result 의 4개의 화합물 정보를 입력으로 각각의 avg_electronegativity 를 구한다.

# In[10]:


avg_electronegativity = []
for index in range(0,len(query_result)):
    avg_electronegativity.append(getElectronegativity_from_dict(query_result.unitcellformula.values[index], atomtable))


# In[11]:


avg_electronegativity


# In[12]:


query_result['avg_electronegativity'] = avg_electronegativity


# 계산된 average electronegativity가 성공적으로 추가되었음을 확인해 볼 수 있다.

# In[13]:


query_result


# 하지만 현재 상태의 query_result는 Machine Learning 의 입력 data로 직접 적용될 수 없다. 이는 lattice, unitcellformula 등의 정보가 dict format 으로 중첩(Nested)되어 있기 때문이다. 중첩된 정보는 pandas 의 tolist() 함수를 이용하여 다수의 독립된 columns 들로 변환할 수 있다. 1) lattice information, 2) unit cell formula 순서로 columns 포멧으로 변환한다.

# 1) Lattice Information: lattice length 에 상응하는 columns 의 수가 작기 때문에 직접 입력해도 무방하다. 과정은 다음과 같다.

# In[14]:


#query_result['lattice'] or query_result.lattice 
#어느 방식을 사용해도 해당 column의 data에 접근할 수 있다.
query_result.lattice


# In[15]:


listed_lattice_length = pd.DataFrame(query_result.lattice.tolist(), columns = ['lattice_a','lattice_b','lattice_c'])
listed_lattice_length


# In[16]:


listed_lattice_angle = pd.DataFrame()
listed_lattice_angle['latticealpha'] = query_result.latticealpha
listed_lattice_angle['latticebeta'] = query_result.latticebeta
listed_lattice_angle['latticegamma'] = query_result.latticegamma
listed_lattice_angle


# In[17]:


listed_lattice_information = pd.concat([listed_lattice_length, listed_lattice_angle], axis = 1)
listed_lattice_information


# 2) Unit Cell Formula: 상응하는 원소의 수가 적지 않으므로 이를 직접 입력하는 방식은 효과적이지 못하다. Reference table의 'symbol' column에 원소 이름의 list가 존재하므로 이를 활용할 수 있다.

# In[18]:


#현재 query result 의 각 화합물 내의 원소 종류 및 원소 개수 확인
query_result.unitcellformula


# In[19]:


#element columns 생성
element_columns = atomtable.symbol.values


# In[20]:


basic_listed_unitcellformula = pd.DataFrame(query_result.unitcellformula.tolist(), columns=element_columns)
basic_listed_unitcellformula.fillna(0, inplace=True)
basic_listed_unitcellformula


# 차후 Machine Learning에 사용할 Label 값들은 per atom 단위의 값이 사용되므로 총원자수 대 해당원자의 비율(rate)값으로 계산한다.

# In[21]:


rate_unitcellformula = []
for row in range(0, query_result.shape[0]):
    rate_unitcellformula.append({k: v / query_result.nsites[row] for k, v in query_result.unitcellformula[row].items()})
query_result['rate_unitcellformula'] = rate_unitcellformula


# In[22]:


query_result


# In[23]:


#element columns 생성
element_columns = atomtable.symbol.values


# 다음과 같이 rate로 나타낸 Formula information 을 얻을 수 있다.

# In[24]:


listed_unitcellformula = pd.DataFrame(query_result.rate_unitcellformula.tolist(), columns=element_columns)
listed_unitcellformula.fillna(0, inplace=True)
listed_unitcellformula


# ####  2.2. 데이터 프레임 병합(Merging)
# 다음으로는 DataType, collectionId, createDate 등 학습에 사용하지 않을 fields 들을 제외한 나머지 features 및 label 정보를 추출한다. 이후 위에서 생성한 lattice information, unitcellformula를 더하여 Machine Learning Datasets을 완성한다.

# In[25]:


#labels and features extraction
listed_extracted = query_result[[
    #Each lable for using Y-value (supervised)
    'bandgap', 'finalenergyperatom','formationenergy', 
    
    #Basic information of each compound 
    'spacegroupnum','nelements', 'nsites','density','mass','volume',
    
    #Derived properties by calculations
    'avg_electronegativity']]


# In[26]:


print("1) listed_extracted: ", listed_extracted.shape)
print("2) listed_lattice_information: ", listed_lattice_information.shape)
print("3) listed_unitcellformula: ", listed_unitcellformula.shape)

for_learning_datasets = pd.concat([listed_extracted, listed_lattice_information], axis = 1)
for_learning_datasets = pd.concat([for_learning_datasets, listed_unitcellformula], axis = 1)

print("--------------------------------------")
print("+) for_learning_datasets: ", for_learning_datasets.shape)

#show the result
for_learning_datasets


# 3종류의 Labels과 125개의 Features를 포함하는 Machine Learning 용 Datasets이 구성되었다.

# ### 3. 심화된 학습 데이터 분석 및 가시화
# 지금까지는 "elements: Li AND nelements:6" 의 조건을 가진 샘플 수준의 데이터로 Dataset building 의 예를 설명하였다. 조금 더 많은 데이터를 추출하여 Dataset을 build 하고, 데이터 간의 상관성을 그래프로 가시화하여 분석하는 기법에 대하여 알아본다.
# #### 3.1. Binary Compounds: 데이터 추출
# 예제의 수행시간을 줄이기 위해 OQMD datasets 중 Binary compounds 데이터를 추출하여 데이터의 상관관계를 분석해본다. 25,877 개의 simulation datasets이 검색됨을 알 수 있다. 

# In[42]:


#쿼리 설정
other_lucene_query = "nelements:2"

full_query = '"'+ server_address + rest_api_option + basic_lucene_query + other_lucene_query + '"'
jsonResultFileName = "query_result_complex.json"

command_line = 'wget -O' + ' ' + currentPath + dataPath + jsonResultFileName +' '+'--user'+ ' '+ user_id + ' '+ '--password' + ' ' + user_pwd + ' ' + extra_args + ' ' + full_query

subprocess.call(command_line, shell=True)

complex_query_result = pd.read_json(currentPath+dataPath+jsonResultFileName)
complex_query_result.fillna(0,inplace=True)


# #### 3.2. Binary Compounds: 학습 데이터 빌드
# 위에서의 예제와 동일한 방식을 이용하여 average electronegativity, lattice information, 그리고 formula information 을 계산한다.

# In[43]:


""" building datasets by adding average electronegativity, converted lattice information, and converted formula
"""
complexset_avg_electronegativity = []
for index in range(0,len(complex_query_result)):
    complexset_avg_electronegativity.append(getElectronegativity_from_dict(complex_query_result.unitcellformula.values[index], atomtable))

# adding average electronegativity
complex_query_result['avg_electronegativity'] = complexset_avg_electronegativity

# converting lattice information
complex_listed_lattice_length = pd.DataFrame(complex_query_result.lattice.tolist(), columns = ['lattice_a','lattice_b','lattice_c'])
complex_listed_lattice_angle = pd.DataFrame()
complex_listed_lattice_angle['latticealpha'] = complex_query_result.latticealpha
complex_listed_lattice_angle['latticebeta'] = complex_query_result.latticebeta
complex_listed_lattice_angle['latticegamma'] = complex_query_result.latticegamma
complex_listed_lattice_information = pd.concat([complex_listed_lattice_length, complex_listed_lattice_angle], axis = 1)


# In[44]:


complex_rate_unitcellformula = []
for row in range(0, complex_query_result.shape[0]):
    complex_rate_unitcellformula.append({k: v / complex_query_result.nsites[row] for k, v in complex_query_result.unitcellformula[row].items()})
complex_query_result['rate_unitcellformula'] = complex_rate_unitcellformula
complex_listed_unitcellformula = pd.DataFrame(complex_query_result.rate_unitcellformula.tolist(), columns=element_columns)
complex_listed_unitcellformula.fillna(0, inplace=True)


# In[45]:


#labels and features extraction
complex_listed_extracted = complex_query_result[[
    #Each lable for using Y-value (supervised)
    'bandgap', 'finalenergyperatom','formationenergy', 
    
    #Basic information of each compound 
    'spacegroupnum','nelements', 'nsites','density','mass','volume',
    
    #Derived properties by calculations
    'avg_electronegativity']]

complex_for_learning_datasets = pd.concat([complex_listed_extracted, complex_listed_lattice_information], axis = 1)
complex_for_learning_datasets = pd.concat([complex_for_learning_datasets, complex_listed_unitcellformula], axis = 1)

print("--------------------------------------")
print("+) for_learning_datasets: ", complex_for_learning_datasets.shape)

#show the result
complex_for_learning_datasets.head()


# In[46]:


complex_for_learning_datasets.describe()


# 한가지 문제를 발견하였다. 만들어진 complex_for_learning_datasets의 데이터 분포 현황에 대하여 데이터 프레임을 describe 하였을 때 columns이 128개에서 127개로 1개 감소하였다. 다음의 코드를 통해 이 문제를 자세히 확인해 볼 수 있다.

# In[47]:


import numpy as np
error_rows = complex_for_learning_datasets[~complex_for_learning_datasets.applymap(np.isreal).all(1)]
error_rows


# In[67]:


error_rows.shape


# 사라진 column은 'bandgap'으로, 3188~18218 rows 의 label 값이 존재하지 않는다. 즉, 위의 47개 simulations들은 bandgap 계산이 이루어지지 않은 단계의 실험이다. 적절한 bandgap 모델을 만들기 위해 해당 rows를 filtering한다. 그리고 난 후 dataset을 float type 으로 casting 한다.

# In[49]:


complex_for_learning_datasets.shape


# In[50]:


error_rows.shape


# In[68]:


filtered_complex_for_learning_datasets = complex_for_learning_datasets[complex_for_learning_datasets.applymap(np.isreal).all(1)]
filtered_complex_for_learning_datasets = filtered_complex_for_learning_datasets.astype(np.float32)
filtered_complex_for_learning_datasets.shape


# In[69]:


filtered_complex_for_learning_datasets.describe()


# 25830 rows의 simulations들이 128개의 columns으로 잘 표현됨을 알 수 있다.

#  <hr/>
# ###### References
# [1] Saal, J. E., Kirklin, S., Aykol, M., Meredig, B., and Wolverton, C. "Materials Design and Discovery with High-Throughput Density Functional Theory: The Open Quantum Materials Database (OQMD)", JOM 65, 1501-1509 (2013). doi:10.1007/s11837-013-0755-4 [Link](http://dx.doi.org/10.1007/s11837-013-0755-4)
# 
#   [2] Kirklin, S., Saal, J.E., Meredig, B., Thompson, A., Doak, J.W., Aykol, M., Rühl, S. and Wolverton, C. "The Open Quantum Materials Database (OQMD): assessing the accuracy of DFT formation energies", npj Computational Materials 1, 15010 (2015). doi:10.1038/npjcompumats.2015.10 [Link](http://dx.doi.org/10.1038/npjcompumats.2015.10)
