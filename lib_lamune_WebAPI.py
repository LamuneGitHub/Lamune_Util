import requests
import json

#---- OpenWeather

"""
get_city_data 함수는 OpenWeather 에서부터 가져온 데이터를 json 형태로
리턴해야 합니다.
"""
def get_city_data(city_name="London,uk"):
    json_data = None

    # API Call
    API_KEY = '0bb4a98b50079bdf526c5a2e518b3362'
    URL_API = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={API_KEY}"
    response = requests.get( URL_API)
    
    # 응답 text를 읽어서 json 객체 생성
    json_data = json.loads( response.text)

    print ( len(json_data) )
    print ( json_data )
    print ( json_data['weather'][0]['description'])
    print ( json_data['main']['temp_min'])

    return json_data



#--- 트위터
import tweepy

def connect_api():
    api = None

    api_key = '8cneIwSOjtxOeRLq488fGfXaW'
    api_key_secret = '7h7zYre2L13YNEBH8ezyeCwHhQWz4kQKDGiaO7QEzjli6UKJe3'
    access_token = '1581927063153348608-YZojjYmKQPfm1Pu2fbZvRtHhiTD5VJ'
    access_token_secret = '924D2XFcDsVhXG9JOQ7HtwPmDxnTfvDfvAc8EcZhx2tLp'

    auth = tweepy.OAuthHandler(api_key, api_key_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    return api

# 트윗 목록 조회
def get_tweets(api, username):
    tweets = []

    # API 사용하여 트윗 받아 오기
    tweet_response = api.user_timeline(username, tweet_mode='extended') # 140 자 이상 가능

    # 받아온 트윗을 반환할 데이터 객체에 적재
    for i, tweet in enumerate(tweet_response):
        tweets.append(f"{i}번째 트윗 : {tweet.full_text}")

    return tweets


import time
import copy
import requests
#--- git hub
def get_github() :
        
    response = requests.get("https://api.github.com/users/octokit/repos")
    # 주의할 점 : github API 는 호출이 너무 많이 발생하면 자체적으로 제한을 걸 수 있습니다.
    # 한번의 API 호출 후 1초 sleep 시간을 지정합니다.
    time.sleep(1)

    if response.status_code == 200:
        octokit = response.json()
        # dont_touch = copy.deepcopy(octokit)   # deep copy 방법

    return octokit


