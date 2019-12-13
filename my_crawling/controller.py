from my_crawling.service import Service
from my_crawling.model import ShopCrawler

class Controller:
    def __init__(self):
        self.service = Service()

    @staticmethod
    def print_menu():
        print('0.EXIT')
        print('1.gsshop')
        print('2.oliveyoung')
        return input('메뉴선택')

    def run(self):
        while 1:
            menu = self.print_menu()
            print('메뉴 : %s' % menu)
            if menu == '1':
                self.service.crawling('gsshop')
            if menu == '2':
                self.service.crawling('oliveyoung')
            elif menu == '0':
                print('BYE')
                break
