# 상필이 작업용 라이브러리
from importlib import reload
from IPython.display import display
import pickle
import time


import pandas as pd
import numpy as np
from numpy import arange


import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns 

#----

#pandas에서 DataFrame을 요약해서 표시하지 않도록 설정
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

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
mpl.rc('font', **font)
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
    y_col_name = score_col_name
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