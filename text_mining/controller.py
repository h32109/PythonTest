from text_mining.service import Service
from my_crawling.model import ShopCrawler

class Controller:
    def __init__(self):
        self.service = Service()

    @staticmethod
    def print_menu():
        print('0.종료\n')
        print('1.파일읽기\n')
        print('2.자연어 처리키트 다운로드\n')
        print('3.토큰처리\n')
        print('4.삭제할 단어보기\n')
        return input('메뉴선택\n')

    def run(self):
        while 1:
            menu = self.print_menu()
            if menu == '1':
                self.service.execute('1')
            elif menu =='2':
                self.service.execute('2')
            elif menu == '3':
                self.service.execute('3')
            elif menu == '4':
                self.service.execute('4')
            elif menu == '5':
                self.service.execute('5')
            elif menu == '0':
                break