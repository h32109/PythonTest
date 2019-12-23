from titanic.controller import Controller
from titanic.view import View
if __name__ == '__main__':
    def print_menu():
        print('0. 종료')
        print('1. 시각화')
        print('2. 정확도체크')
        print('3. 교차검증')
        print('4. 모델추출')
        return input('메뉴 입력\n')

    while 1:
        menu = print_menu()
        if menu == '1':
            app = Controller()
            train = app.create_model('train.csv')
            print(train)
            vue = View(train)
            menu = input('차트 내용 선택\n'
                         '1. 생존자 vs 사망자\n'
                         '2. 생존자 성별 대비')
            if menu == '1':
                vue.plot_survived_dead()
            elif menu == '2':
                vue.plot_sex()
        if menu == '2':
            pass
        elif menu == '0':
            break