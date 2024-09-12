import requests
from bs4 import BeautifulSoup

url = "https://research.google.com/youtube8m/explore.html"

page = requests.get(url)
soup = BeautifulSoup(page.text, 'html.parser')

youtube_links_div = soup.find('div', id="thumbs")
number_of_links = 0

anchor_tags = youtube_links_div.find_all('a')

youtube_links = [link.get('href') for link in anchor_tags]

print(youtube_links_div)
print(len(youtube_links))
