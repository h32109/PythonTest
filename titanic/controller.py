from titanic.service import Service
from titanic.view import View

class Controller:
    def __init__(self):
        self.service = Service()
        self.view = View()
        self._list = []


    @property
    def list(self) -> object:
        return self._list


    @list.setter
    def list(self, list):
        self._list = list


    def print_menu(self):
        print('0. 종료')
        print('1. 데이터로딩')
        print('2. 차원축소 ')
        print('3. Nominal 편집')
        print('4. 시각화')
        return input('메뉴 입력\n')


    def run(self):


        while 1:
            menu = self.print_menu()
            if menu == '1':
                self.list = self.service.load_data()
            if menu == '2':
                feature = input('축소할 차원 입력\n') #Cabin, Ticket
                self.list = self.service.drop_feature(self.list, feature)
            if menu == '3':
                self.list = self.service.sex_nominal(self.list)
            if menu == '4':
                self.list = self.service.load_data()
                view = input('차트 내용 선택\n'
                             '1. 생존자 vs 사망자\n'
                             '2. 생존자 성별 대비')
                if view == '1':
                    self.view.showPlot(self.list)
                elif view == '2':
                    self.view.plot_sex(self.list)
            elif menu == '0':
                break
