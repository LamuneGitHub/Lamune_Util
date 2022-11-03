from selenium import webdriver
from selenium.common.exceptions import WebDriverException as WDE
from selenium.webdriver.common.keys import Keys
import requests
from bs4 import BeautifulSoup
import time

# 셀레니움 객체 준비
# path = 'C:/Users/admin/Desktop/Pythonworkspace/chromedriver_win32/chromedriver.exe'
path = "/Users/lamune/git/git_hub/ds-sa-star-scraper/src/chromedriver"
browser = webdriver.Chrome(path) 
browser.maximize_window()

# url 접속 
url = "https://youtube.com/"
browser.get(url)
time.sleep(2)
present_url = browser.current_url
browser.get(present_url)


#(TODO) 로그인은 어떻게 하나?

# 스크롤을 어디까지 내리는지 기준 
# finish_line = 40000 기준: 162 개
finish_line = 1000
last_page_height = browser.execute_script("return document.documentElement.scrollHeight")

while True:
    # 우선 스크롤 내리기
    browser.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
    time.sleep(2.0)       # 작업 중간에 1이상으로 간격을 줘야 데이터 취득가능(스크롤을 내릴 때의 데이터 로딩 시간 때문)
    # 현재 위치 담기
    new_page_height = browser.execute_script("return document.documentElement.scrollHeight")
    
    # 과거의 길이와 현재 위치 비교하기
    if new_page_height > finish_line:
        break
    else: 
        last_page_height = new_page_height
