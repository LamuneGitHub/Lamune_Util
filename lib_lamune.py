"""
# 라이브러리 재 로딩

import lib_lamune as lmn
lmn.reload(lmn)

"""

"""
# 구글 드라이브 mount
from google.colab import drive
drive.mount('/content/drive')
"""


#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
# 라이브러리 로딩

import sys

# if "google.colab" in sys.modules:
#     # Install packages in Colab
#     !pip install category_encoders==2.*
#     !pip install eli5
#     !pip install pandas-profiling==2.*
#     !pip install pdpbox
#     !pip install --upgrade xgboost

# # !conda install -c conda-forge pdpbox

import warnings
warnings.filterwarnings("ignore")


# 상필이 작업용 라이브러리
from importlib import reload
import pickle
import time

from IPython.display import display

import re
import math
import random
import numpy as np
from numpy import arange
import pandas as pd
#pandas에서 DataFrame을 요약해서 표시하지 않도록 설정
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# 시각화 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# AI 
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
from scipy.stats import ttest_1samp, ttest_ind
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# xgboost.config.set_config(verbosity=0)

#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
# 기본 함수 정의

# 줄 구분 표시 출력 
def print_line() :
  print ("\n------------------------------------------------")
def print_line_m() :
  print ("\n-----------------------")
def print_line_s() :
  print ("\n----------")

# 기본 함수 정의
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
# EDA 함수



# 결측치가 포함된 df를 반환 한다. ( copy 된 data)
"""
    str_replace = np.nan # 결측치를 변경할 문자
    ,need_replace = False # 결측치 변경이 필요한지 여부 
    ,need_drop = False # 결측치 행의 삭제가 필요한지 여부
"""
def 결측치행열조회( df_target , str_replace = np.nan , need_replace = False , need_drop = False) :

    # 결측치로 인식할 문자열
    #str_replace = '-' # 결측치를 변경할 문자
    lst_na_str = [ np.NaN , '-' , "N/A" 'NA' , 'na', 'NaN' , "" ]   # 결측치로 인식할 문자들 

    # 조회
    cond_tmp1 = df_target.isin(lst_na_str).any( axis= 1)

    # isnull()의 결과는 위에 포함 되므로 반복 처리 해 주지 않아도 된다. 
    """
    # 빈값 조회
    cond_tmp2 = df_target.isnull().any(axis= 1)

    # 빈값 이거나, 빈값 문자열인 경우
    cond_tmp3 = cond_tmp1 | cond_tmp2

    set_1 = set(trueIndex리스트반환( cond_tmp1 ))
    set_2 = set(trueIndex리스트반환( cond_tmp2 ))

    print ( set_1 > set_2 )
    print ( set_1 == set_2 )
    print ( set_1 < set_2 )"""

    # 결측치가 포함된 컬럼명 조회
    sri_tmp = df_target.isin(lst_na_str).any()
    list_col = sri_tmp[sri_tmp == True].index

    # 작업 Df 생성
    df_tmp = df_target[cond_tmp1][list_col].copy()
    #display ( df_tmp  )
    #display ( df_target[cond_tmp1][list_col]  )

    # 출력
    print ( f"결측치가 포함된 Data 갯수 {len(df_tmp)} / { len(df_target)} " )
    print ( f"각 컬럼별 결측치 갯수")
    print ( f"{df_tmp.isin(lst_na_str).sum()}")

    # (TODO)결측치로 인식한 문자열을  str_replace 으로 변경
    if need_replace :
        df_tmp.replace( np.nan , str_replace , inplace=True)

    # 결측치 행 삭제
    if need_drop :
        df_target.drop(df_tmp.index , inplace=True)

    return df_tmp


# EDA 기본정보 자동 분석 함수
def eda_info (df_param, need_category = False , category_count = 20) :

  df_target = df_param
  
  # df 변형이 필요한 경우 copy 
  if need_category :
    df_target = df_param.copy()
  
  list_columns = df_target.columns

  # 데이터셋 정보 확인
  #: 컬럼 자료형 , 전체 자료 갯수 , 컬럼 갯수 , 결측치 여부 , 전체 data 크기
  print_line()
  print ( "# 데이터셋 정보 확인" )
  print ( df_target.info() )


  # 컬럼 자료형 조회 (dtypes 이용) 
  #df_titanic.dtypes


  # 데이터의 행열 크기 조회
  print_line()
  print( "# 데이터의 행열 크기 = " , df_target.shape )


  # 결측치가 존재하는 컬럼명 조회
  print_line()
  """
  print( "# 컬럼별 결측치 갯수 ")
  print( df_target.isnull().sum() )

  print_line()
  print( "# 결측치 총 갯수 = " , df_target.isnull().sum().sum())
  """
  df_null = 결측치행열조회( df_target )
  print ( df_null.head(print_row_count) )


  # 컬럼별 중복값 확인
  print_line()
  print( "# 컬럼별 중복값 확인 ")
  for tmp_column in list_columns :
    # 중복값 갯수 조회
    tmp_dup_count = df_target[tmp_column].duplicated(keep = False).sum()
    
    if tmp_dup_count > 0 :
      print ( tmp_column , "컬럼에 중복값", tmp_dup_count , "개" )
      #print ( "index = " , df_target[tmp_column].duplicated(keep = False)   )
      #TODO : 중복값 인덱스 출력
    
  # 중복 데이터 확인
  print( "중복 data 갯수(전체)" , df_target.duplicated(keep = False).sum() )
  print( "중복 data " , df_target[df_target.duplicated(keep = False)].head(print_row_count) )


  # 각 컬럼별 값의 비율 조회
  # 컬럼명 조회
  print_line()
  print( "# 컬럼별 값의 비율 ")
  print ("컬럼 목록= " , list_columns)

  # 컬럼별 값의 비율 조회
  list_category_col_name = [] # 카테고리화 가능할 것으로 예상되는 컬럼명
  for tmp_column in list_columns :
    df_temp = df_target[tmp_column].value_counts(normalize=True ,dropna = True)
    
    print_line_s()
    print ( f"컬럼명 = {tmp_column} , 값의종류 갯수 = {len(df_temp)} " )
    
    if len(df_temp) <= category_count :
      list_category_col_name.append( tmp_column )
      print ( "Category 가능성 있음")
      print ( df_temp )
    #else :
      #print ( "값의 종류가 20개 이상임" )


  # 카테고리화 가능한 컬럼은 카테고리화 진행 
  if need_category :
    print ( "카테고리화 진행")
    for col_name in list_category_col_name :
      df_target[col_name] = df_target[col_name].astype('category')


  # 컬럼별 통계치 확인
  print_line()
  print ( "#컬럼별 통계특성 확인" )
  print ( df_target.describe())


  # 컬럼별 포함된 큭수문자 의 종류 조회
  print_line()
  print( "# 컬럼별 포함된 특수문자 종류 ")
  regx_특수문자제외한 = r'([^0-9a-zA-Z]+)'
  for tmp_column in list_columns :

    if df_target[tmp_column].dtype == 'object'  :
      print ( "컬럼명 = " , tmp_column )
      
      df_rslt = df_target[tmp_column].str.extractall( r'([^0-9a-zA-Z]+)'  )
      #display ( type(df_rslt) ) #=> DataFrame 

      set_rslt = []
      for idx in df_target.index :
        
        # TODO 값이 Nan인 경우의 정규표현식을 조사하지 않도록 한다,
        # 우선은 str인 경우만 
        tmp_val = df_target.loc[ idx , tmp_column]
        if isinstance( tmp_val , str) : 
          rslt = re.findall ( regx_특수문자제외한 , df_target.loc[ idx , tmp_column] )
          set_rslt += rslt 
        #else :
        #  display ( tmp_val )

      print( set(set_rslt) )
      
      print_line_s()


  # category 컬럼의 각 값 별 갯수와 비율 조회
  print_line()
  print ( "#category 타입 컬럼의 각 값 갯수와 비율 조회" )
  list_category_col_name = df_target.dtypes[df_target.dtypes.values == 'category'].index.tolist() 
  for col_name in list_category_col_name :
    sri_a = df_target[col_name].value_counts(dropna = True)
    sri_b = df_target[col_name].value_counts(normalize=True ,dropna = True)
    df_a = sri_a.to_frame( name = 'count')
    df_b = sri_b.to_frame( name = 'ratio')

    print (f"컬럼명 = {col_name} ")
    print ( df_a.join( df_b ) )
    print_line_s()

  #TODO 컬럼별 값 분포 그래프 표시
  #Boxplot과 Histogram을 통해서도 이상치 존재 여부를 확인해볼 수 있습니다.

  return df_target

# 숫자와 . 이외에 어떤 문자가 들어가 있는지 확인 하고 문제가 있는 Data를 df 형식으로 반환
# 
def 숫자이외의문자확인 (df_target , list_columns ) :
  list_err_idx = []
  regx_숫자를제외한 = r'([^0-9.]+)'
  for tmp_column in list_columns :

    if df_target[tmp_column].dtype == 'object'  :
      print ( "컬럼명 = " , tmp_column )
      
      df_rslt = df_target[tmp_column].str.extractall( regx_숫자를제외한  )
      #display ( type(df_rslt) ) #=> DataFrame 

      list_err_str = [] 
      for idx in df_target.index :
        tmp_val = df_target.loc[ idx , tmp_column]

        # TODO 값이 Nan인 경우의 정규표현식을 조사하지 않도록 하는 조건 추가
        # 우선은 str인 경우만 체크 진행 하도록 하였음
        if isinstance( tmp_val , str) : 

          rslt = re.findall ( regx_숫자를제외한 , df_target.loc[ idx , tmp_column] )
          if len(rslt) != 0 :
            list_err_str += rslt
            list_err_idx.append( idx )

        #else :
        #  display ( tmp_val )

      print( set(list_err_str) )
      
      print_line_s()

  err_idx = sorted( set(list_err_idx)) 

  # err 데이터 반환
  return df_target.loc[err_idx]

# DataFrame의 특정 컬럼의 data중에서 cond_sel 조건에 맞는 data를 fix_func 으로 수정 후 수정된 DataFrame을 반환
def fix_column_err_value ( df_temp_clean: pd.DataFrame , target_column: '수정할 column 명' 
                          , cond_sel:'수정 데이터 check 조건' 
                          , fix_func:'err를 수정할 함수' 
                          ) -> pd.DataFrame : #'수정된 DataFrame' 

    # err 조건에 해당하는 data 조회
    sri_err = df_temp_clean[ target_column ].apply( cond_sel )

    # 조회 결과 확인 
    display ( "수정전")
    display ( sri_err.sum() )
    display ( df_temp_clean.loc[ sri_err ].head(print_row_count) )


    # 오류를 수정 후 data 확인
    #(TODO) sri_fixed_col = df_temp_clean.loc[ sri_err ][[target_column]].apply( fix_func ,axis=1)
    sri_fixed_col = df_temp_clean.loc[ sri_err ][target_column].apply( fix_func )
    #display (sri_fixed_col.head(3))

    # 오류 수정한 데이터를 전처리 작업용 DataFrame에 반영
    df_temp_clean.loc[ sri_err, target_column ] = sri_fixed_col

    # 반영 결과 data 확인 
    display ( "반영결과")
    display ( df_temp_clean[sri_err].head(print_row_count))

    # # 반영 결과 집계 확인 
    # sri_err = df_temp_clean[ target_column ].apply( cond_sel )
    # display ( sri_err.sum() )

    return df_temp_clean

# EDA 함수
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------
# 피쳐 엔지니어링


# 컬럼명의 suffix 인 컬럼을 제거
#
def FE_컬럼명_drop_y(df, suffix= '_yy'):
    # list comprehension of the cols that end with '_y'
    to_drop = [x for x in df if x.endswith(suffix)]
    df.drop(to_drop, axis=1, inplace=True)


# 중복컬럼 없이 merge
#
def FE_merge_중복컬럼제거 ( *param, **param2 ) :
  df_temp = pd.merge( *param , **param2 , suffixes=('', '_yy'))
  
  # 중복컬럼명 조회
  lst_cols = df_temp.columns
  lst_dup_cols_name = []
  for col_name in lst_cols :
    if col_name.endswith('_yy') :
      org_col_name = col_name[:-3]
      lst_dup_cols_name.append(org_col_name)

  # 빈값을 머지한 컬럼의 값으로 변경
  for col_name in lst_dup_cols_name :
    df_temp.loc[df_temp[col_name].isnull() ,col_name ] = df_temp.loc[df_temp[col_name].isnull() , col_name+'_yy']

  # _yy 중복 컬럼을 삭제  
  FE_컬럼명_drop_y( df_temp ,'_yy')
  
  return df_temp


def FE_컬럼앞에_구분명_추가 ( df_tmp , prefix ) :
  
  lst_columns_from = df_tmp.columns
  lst_columsn_tobe = []
  for col_name in lst_columns_from :
      col_name_tmp = col_name.replace( " " , "_")
      lst_columsn_tobe.append(  prefix + '_' +  col_name_tmp  )

  df_tmp.columns = lst_columsn_tobe

  return df_tmp


def FE_날짜컬럼_추가( df_tmp , col_date_name_from='변환' , replace_index = False , type='D') :

    if type == 'D' :

        # 날짜 컬럼을 추가
        #

        # 날짜 컬럼을 맨 앞으로 순서변경 하기 위한 준비
        lst_col_name = df_tmp.columns.to_list()
        lst_col_name_dt = ['DT_date' , 'DT_Year', 'DT_Month' , 'DT_Day' , 'DT_DayOfWeek' , 'DT_DayOfYear' ]

        # 날짜 값 컬럼 추가
        col_date_name_to = 'DT_date'
        df_tmp[col_date_name_to] = pd.to_datetime(df_tmp[col_date_name_from])
        df_tmp['DT_Year'] = df_tmp[col_date_name_to] .dt.year
        df_tmp['DT_Month'] = df_tmp[col_date_name_to] .dt.month
        df_tmp['DT_Day'] = df_tmp[col_date_name_to] .dt.day
        df_tmp['DT_DayOfWeek'] = df_tmp[col_date_name_to] .dt.day_of_week
        df_tmp['DT_DayOfYear'] = df_tmp[col_date_name_to] .dt.day_of_year
        
        # 날짜 컬럼을 맨 앞으로 순서변경
        df_tmp =df_tmp[ lst_col_name_dt + lst_col_name ]
        df_tmp.drop( columns=[col_date_name_from] ,inplace=True)

        # 인덱스를 날짜 값으로 변경
        if replace_index :
            df_tmp.index = df_tmp[col_date_name_to]

    return df_tmp 

# 피쳐 엔지니어링
#-----------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------


#----

#pandas에서 DataFrame을 요약해서 표시하지 않도록 설정
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

#----
# 그래프 한글 출력 깨짐 해결
# import matplotlib.font_manager
# import matplotlib.pyplot as plt
# import matplotlib as mpl

# [f.name for f in matplotlib.font_manager.fontManager.ttflist if 'Nanum' in f.name]
[f.name for f in matplotlib.font_manager.fontManager.ttflist if 'D2' in f.name]

plt.style.use('ggplot')
font = {'size': 12,
        #'family': 'NanumBarunGothic'}
        'family': 'D2Coding'}
matplotlib.rc('font', **font)
#----- 





#----


# 하이퍼 파라미터 랜덤 튜닝 결과 저장 (with pickle)
def dict파일저장( dict_save , v_file_name = '하이퍼파라미터_튜닝' ) :

    import pickle
    import time

    # 저장할 파일명 지정
    now = time
    file_name = v_file_name + "_" + now.strftime('%Y_%m_%d__%H%M%S_%s') + ".pickle"

    # 하이퍼 파라미터 튜닝결과를 파일로 저장
    with open( file_name ,'wb') as fw:
        pickle.dump( dict_save , fw)

    print ( f"파일명 : {file_name} ")

    return file_name


# dict 파일 로드
def dict파일로드 ( v_file_name ) :

    import pickle
    import time

    # 피클 데이터 로딩

    with open( v_file_name , 'rb') as fr:
        user_loaded = pickle.load(fr)

    print ( f"dict_  {  user_loaded } ")

    return user_loaded



#-------

# 최적 값에 대한 X_val data의 성능지표 및 기록
# ret =  dict_성능지표
#
def 성능지표계산 ( model, x_train, y_train, x_test, y_test , need_train = True  ):

    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

    dict_성능지표 = {} 

    if need_train :
        # 모델 학습
        model.fit(x_train, y_train)

    # 학습 데이터
    y_pred = model.predict(x_train)
    y_data = y_train
    train_accuracy_score = accuracy_score(y_data, y_pred) # 정확도
    train_precision_score = precision_score(y_data, y_pred) # 정밀도
    train_recall_score = recall_score(y_data, y_pred) # 재현율
    train_f1_score = f1_score(y_data, y_pred) # F1 스코어

    # print(f"학습 Accuracy:   {train_accuracy_score:.3f}") # 정확도
    # print(f"학습 Precision:  {train_precision_score:.3f}") # 정밀도
    # print(f"학습 Recall:     {train_recall_score:.3f}") # 재현율
    # print(f"학습 F1-score:   {train_f1_score:.3f}") # F1 스코어

    dict_성능지표['train_accuracy_score'] = train_accuracy_score
    dict_성능지표['train_precision_score'] = train_precision_score
    dict_성능지표['train_recall_score'] = train_recall_score
    dict_성능지표['train_f1_score'] = train_f1_score


    # 테스트 데이터
    y_pred = model.predict(x_test)
    y_data = y_test
    test_accuracy_score = accuracy_score(y_data, y_pred) # 정확도
    test_precision_score = precision_score(y_data, y_pred) # 정밀도
    test_recall_score = recall_score(y_data, y_pred) # 재현율
    test_f1_score = f1_score(y_data, y_pred) # F1 스코어

    # print(f"테스트 Accuracy:   {test_accuracy_score:.3f}") # 정확도
    # print(f"테스트 Precision:  {test_precision_score:.3f}") # 정밀도
    # print(f"테스트 Recall:     {test_recall_score:.3f}") # 재현율
    # print(f"테스트 F1-score:   {test_f1_score:.3f}") # F1 스코어

    dict_성능지표['test_accuracy_score'] = test_accuracy_score
    dict_성능지표['test_precision_score'] = test_precision_score
    dict_성능지표['test_recall_score'] = test_recall_score
    dict_성능지표['test_f1_score'] = test_f1_score

    print ( f"dict_성능지표  {  dict_성능지표 } ")

    return dict_성능지표



### # 테스트 성능 결과표 & 그래프 그리기
# graph_col_count = 3 # 한줄의 컬럼 갯수
def 랜덤_하이퍼파라미터_튜닝결과_그래프그리기 (cv_results_ , graph_col_count = 3, text_font_size=10 , detail_depth = 5) :

    import matplotlib
    import seaborn as sns 
    import matplotlib.pyplot as plt


    # score_col_name = 'rank_test_f1'
    rank_col_name = 'rank_test_score'
    score_col_name = 'mean_test_score'

    #테스트 성능 결과표 
    df_cv_results_ = pd.DataFrame(cv_results_)


    lst_cols_all = df_cv_results_.columns
    lst_colname = [ col for col in lst_cols_all if ("param_" in col) or ("mean_" in col) or ("rank_" in col)]
    lst_colname = sorted(lst_colname)
    df_tunning_rslt = df_cv_results_.sort_values(by=rank_col_name)[lst_colname]
    # display (df_tunning_rslt.T )


    # seaborn 으로 그래프를 그리기 위한 전처리
    #   0 부터 시작하는 인덱스 컬럼을 생성
    df_tmp = df_tunning_rslt
    df_tmp["sns_x_index"] = df_tmp[rank_col_name] -1

    display (df_tmp.T )

    # 출력할 컬럼의 목록 정의
    lst_param_colname = [ col for col in lst_cols_all if "param_" in col ]
    lst_param_colname = sorted(lst_param_colname)

    print ( len(lst_param_colname) , lst_param_colname )


    # 그래프 크기 계산
    # graph_col_count = 4 # 한줄의 컬럼 갯수
    graph_row_count = round( ( 1 + len(lst_param_colname) ) / graph_col_count ) # 필요한 열 갯수
    figsize = ( graph_col_count * 6 , graph_row_count * 5) # 그래프 크기 계산
    fig, axes = plt.subplots(graph_row_count, graph_col_count, figsize=figsize) # 그래프 영역, 설정
    # matplotlib.rc_file_defaults()
    # ax1 = sns.set_style(style=None, rc=None )
    #fig, ax1 = plt.subplots(figsize=(12,6))


    #-------------
    # 그래프 1
    # 성능 vs 시간  비교 그래프 
    #
    ax=axes[0,0]

    # 계산 시간 막대 그래프
    x_col_name = "sns_x_index"
    y_col_name = "mean_fit_time"
    ax1 = sns.barplot ( data=df_tmp , x=x_col_name , y=y_col_name , label=y_col_name
        , color='b' , alpha=0.2 
        , ax=ax
    ) #, ax=ax1)
    plt.ylim ( 0, 100 )

    # 성능지표 선 그래프
    ax2 = ax1.twinx()
    y_col_name = score_col_name   # "mean_test_f1"
    ax2 = sns.lineplot ( data=df_tmp , x=x_col_name , y=y_col_name , label=y_col_name
        , alpha=0.2 , marker='o' #, color='b'#, sort = False 
        , ax=ax2)

    # y_col_name = "mean_test_roc_auc"
    # ax2 = sns.lineplot ( data=df_tmp , x=x_col_name , y=y_col_name , label=y_col_name
    #     , alpha=0.2 , marker='o' #, color='b'#, sort = False 
    #     , ax=ax2)


    # 숫자 넣는 부분, height + 0.25로 숫자 약간 위로 위치하게 조정
    y_col_name = score_col_name
    for x , y in zip( df_tmp[x_col_name] , df_tmp[y_col_name] ) :
        plt.text(x, y,   round( y, 3)
        , ha='center', va='bottom', size = text_font_size, color = 'black')

    plt.ylim ( 0.5, 0.9 ) 

    #(v)) 선 그래프와 막대 그래프의 X축 위치가 안 맞는 문제
    #(v) 시본에서 막대그래프 + 선 그래프 출력시 라인그래프 밀리는 증상 수정
    ax2.set(xticks=range(10), xticklabels=list(range(1,11)))

    title = "성능지표 vs 시간" 
    ax.set_title(title)


    #-------------
    # 그래프 2~
    # 
    # 각 파라미터 값 그래프 
    # 
    for i, col_name in enumerate(lst_param_colname):
        i = i +1
        ax = axes[ i // graph_col_count , i % graph_col_count]

        x_col_name = "sns_x_index"
        y_col_name = col_name
        
        ax = sns.lineplot ( data=df_tmp , x=x_col_name , y=y_col_name #, label=y_col_name
            , alpha=0.2 , marker='o' #, color='b'#, sort = False 
            , ax=ax)

        # 숫자 넣는 부분, height + 약간 위로 위치하게 조정
        for x , y in zip( df_tmp[x_col_name] , df_tmp[y_col_name] ) :
            ax.text( x, y,  round( y , detail_depth)
            # ax.text( x, y,  y
            , ha='center', va='bottom', size = text_font_size, color = 'black' )

        #(v) 시본에서 막대그래프 + 선 그래프 출력시 라인그래프 밀리는 증상 수정
        # 0 부터 시작하는 인덱스 1 부터 표시 하도록 수정
        ax.set(xticks=range(10), xticklabels=list(range(1,11)))

        # 그래프 제목
        title = col_name.replace ("param_xgbclassifier__" , "")
        ax.set_title(title)
        # axes[i//4, i%4].legend()


    plt.tight_layout()
    plt.show()


# 하이이퍼라미터 튜팅용 조건을 좁히기 ( Random )
def 파라미터조건좁히기( dct_하이퍼파라미터_튜닝결과 , n_count , detail_level ,  성능기준_파라미터 = 'rank_test_f1' ) :

    # 출력할 컬럼의 목록 정의
    df_하이퍼파라미터_튜닝결과_Rank_sorted = pd.DataFrame(dct_하이퍼파라미터_튜닝결과).sort_values(by=성능기준_파라미터)
    # display ( df_cv_results_.T )

    lst_cols_all = df_하이퍼파라미터_튜닝결과_Rank_sorted.columns
    lst_param_colname = sorted( [ col for col in lst_cols_all if "param_" in col ] )

    print ( f" {len (lst_param_colname)} cols = { lst_param_colname }" )

    param_tmp ={}
    for col_name in lst_param_colname : 

        key = col_name.replace("param_" , "" )
        min = round( df_하이퍼파라미터_튜닝결과_Rank_sorted[0:n_count][col_name].min() , detail_level )
        max = round( df_하이퍼파라미터_튜닝결과_Rank_sorted[0:n_count][col_name].max() , detail_level )

        # 파라미터값의 min과 max가 같은 경우
        if min == max :
            print ( f" {key}  = {min} : 최적화 완료 ")
            param_tmp[key] = [min]
            continue 

        # 정수인 경우
        step = None
        if (int(min) == min) and (int(max) == max) :
            step = round( (max - min ) / 10  )    
        else :
            # 실수 인경우
            step = round( (max - min ) / 10 , detail_level + 1 )

        # 딕셔너리에 추가
        # print ( f" {key} min = {min} max = {max} step= {step} ")
        param_tmp[key] = [x for x in arange( min , max , step)]

    print ( f"param len = { len( param_tmp )} param = { param_tmp }")

    return param_tmp    

# 하이퍼파라미터_튜닝결과_성능지표_저장
def 하이퍼파라미터_튜닝결과_성능지표_저장 (dict_성능지표 , df_하이퍼파라미터_튜닝결과_select ,  df_성능지표_hist  ) :  

    # 저장할 데이터 
    # dict_성능지표 + 하이퍼파라미터 튜닝 결과 best의 파라미터 정보

    # 하이퍼파라미터 튜닝 결과 best의 파라미터 정보
    lst_colname = sorted( [ col for col in df_하이퍼파라미터_튜닝결과_select.columns if ("param_" in col) or ("mean_" in col) or ("rank_" in col)] )
    dict_best_튜닝정보 = df_하이퍼파라미터_튜닝결과_select[lst_colname].to_dict()

    # dict_성능지표 + dict_best_튜닝정보
    dict_성능지표.update( dict_best_튜닝정보 )

    df_성능지표_hist = pd.concat( [df_성능지표_hist , pd.DataFrame( dict_성능지표)] , axis=0  ,ignore_index=True)

    # 파일에 저장
    #

    # 저장할 파일명 지정
    now = time
    file_name = "df_성능지표_hist_" + now.strftime('%Y_%m_%d__%H%M%S_%s') + ".csv"
    df_성능지표_hist.to_csv(file_name, index=False)

    print ( f"파일명 : {file_name} ")

    return df_성능지표_hist



