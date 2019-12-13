from bs4 import BeautifulSoup
from urllib.request import urlopen
class ShopCrawler:
    def __init__(self, url):
        self.url = url

    def scrap(self):
        soup = BeautifulSoup(urlopen(self.url), 'html.parser')
        for i in soup.find_all(name='p', attrs=({'class': 'tx_name'})):
            print('상품: '+i.text)
        print('=============================')
