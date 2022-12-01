import re
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://movie.naver.com/movie"

"""
get_page 함수는 페이지 URL 을 받아 해당 페이지를 가져오고 파싱한 두
결과들을 리턴합니다.

예를 들어 `page_url` 이 `https://github.com` 으로 주어진다면
    1. 해당 페이지를 requests 라이브러리를 통해서 가져오고 해당 response 객체를 page 변수로 저장
    2. 1번의 response 의 html 을 BeautifulSoup 으로 파싱한 soup 객체를 soup 변수로 저장
    3. 저장한 soup, page 들을 리턴하고 함수 종료

파라미터:
    - page_url: 받아올 페이지 url 정보입니다.

리턴:
    - soup: BeautifulSoup 으로 파싱한 객체
    - page: requests 을 통해 받은 페이지 (requests 에서 사용하는 response
    객체입니다).
"""
def get_page(page_url):
    soup =None
    page = None


    response = requests.get( page_url  )

    if response.status_code != 200  :
        print ( f"접속 실패{response.status_code} ")
        return

    # print( response.url)
    soup = BeautifulSoup(response.text, 'html.parser')

    page = response
    return soup, page



# soup 사용 예
def get_예제(movie_title ):

    search_url = f"{BASE_URL}/search/result.naver?query={movie_title}&section=all&ie=utf8"
    # print( f"search_url = {search_url}")

    soup, page = get_page (search_url)
    # print ( f"page = {page.text}")

    # case 01
    # 속성값 조회
    find_result = soup.select( ".search_list_1 > li > .result_thumb > a" )[0]["href"].split("=")[1]
    # print ( f"find_result = {find_result}" )

    # clase 02
    # tag text 조회
    find_result = soup.select( ".star_score > em ")
    lst_score = [ int(x.getText()) for x in find_result ]



        
    movie_code = int(find_result)

    return movie_code



# 예제 영화 코드 조회
def get_movie_code(movie_title):
    search_url = f"{BASE_URL}/search/result.naver?query={movie_title}&section=all&ie=utf8"
    soup, page = get_page (search_url)

    find_result = soup.select( ".search_list_1 > li > .result_thumb > a" )[0]["href"].split("=")[1]
    movie_code = int(find_result)

    return movie_code


# Lamune
# 페이지 에서 지정된 객체를 read_cnt 갯수 만큼만 반환
def get_data( read_cnt , search_url ) :
    # print ( f"read_cnt={read_cnt}, search_url = {search_url} ")

    reviews = []

    # URL 접속 하여 html 회득
    # print( f"search_url = {search_url}")
    soup, page = get_page (search_url)
    # print ( f"page = {page.text}")

    # 점수 data 가져오기
    find_result = soup.select( ".star_score > em ")
    lst_score = [ int(x.getText()) for x in find_result ]

    # 리뷰 data 가져오기
    find_result = soup.select( ".score_reple ")
    
    lst_review = []
    for itm in find_result :
        a = itm.select ( "p > span" )
        b=None
        if len(a) == 1 :
            b = a[0]
        else : 
            b = a[1]

        lst_review.append (b.getText().replace('\t',"").replace('\r',"").replace('\n',""))

        # try :
        #     lst_review.append (a[1])
        # except :
        #     print ( f" itm = { a} ")
        #     print ( f" a[0] = { a[0]} ")
        #     print ( f" a[1] = { a[1]} ")
                

    # print ( lst_review )
    # lst_review = [ x.select ( "p > span" )[1].getText().replace('\t',"").replace('\r',"").replace('\n',"") for x in find_result ]

    # 지정된 갯수만큼 데이터 적재
    for idx in range ( len(find_result) ):

        # 짜투리 갯수 만큼만 추가
        itm_cnt = idx + 1
        if itm_cnt > read_cnt :
            break
        itm_cnt +=1

        dict_tmp = dict()
        dict_tmp['review_text'] = lst_review[idx]
        dict_tmp['review_star'] = lst_score[idx]

        reviews.append ( dict_tmp )
    
    return reviews

# 페이지 순회 예제
def scrape_by_review_num(movie_title, review_num):
    # 반환 구조체 정의
    reviews = []

    # 필요한 페이지 수 계산 
    페이지당_아이템_갯수 = 10
    import math
    need_page_cnt = math.ceil( review_num / 페이지당_아이템_갯수 )
    if review_num % 페이지당_아이템_갯수  == 0 :
        짜투리 = 페이지당_아이템_갯수
    else : 
        짜투리 = review_num % 페이지당_아이템_갯수 


    #영화code 조회
    movie_code = get_movie_code(movie_title)


    #URL 바꿔 가며 페이지 순회 
    page = 1
    last_page = need_page_cnt
    for page in range ( 1 , last_page + 1 , 1 ) :
        search_url = f"https://movie.naver.com/movie/bi/mi/pointWriteFormList.naver?code={movie_code}&page={page}"

        read_cnt = 페이지당_아이템_갯수
        if page == last_page :
            read_cnt = 짜투리
        reviews += get_data( read_cnt , search_url ) 

    # print ( reviews ) 
    return reviews