import os
import urllib.request
import json
from dotenv import load_dotenv


def fetch_links_from_naver_api(query, api_url, top_n):
    client_id = os.getenv('NAVER_CLIENT_ID')
    client_secret = os.getenv('NAVER_CLIENT_SECRET')

    if not client_id or not client_secret:
        raise ValueError("Environment variables NAVER_CLIENT_ID or NAVER_CLIENT_SECRET are missing")

    enc_text = urllib.parse.quote(query)
    url = f"{api_url}?query={enc_text}"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)

    try:
        response = urllib.request.urlopen(request)
        rescode = response.getcode()
        if rescode == 200:
            response_body = response.read()
            data = json.loads(response_body)
            links = [item['link'] for item in data['items'][:top_n]]
            return links
        else:
            print("Error Code:" + str(rescode))
            return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def search_naver(query, top_n=20):
    # .env 파일에서 환경 변수 로드
    load_dotenv()

    web_links = fetch_links_from_naver_api(query, "https://openapi.naver.com/v1/search/webkr.json", top_n)
    news_links = fetch_links_from_naver_api(query, "https://openapi.naver.com/v1/search/news.json", top_n)

    return web_links + news_links


# 함수 사용 예제
if __name__ == "__main__":
    query = "알라딘"  # 검색어를 원하는 대로 변경
    links = search_naver(query)
    print(links)
