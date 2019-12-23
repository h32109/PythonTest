from kbstar.controller import Controller
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
            train = app.create_model('kbstar.xls')
            print(train)

        if menu == '2':
            pass
        elif menu == '0':
            break
