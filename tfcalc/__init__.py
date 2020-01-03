import tensorflow as tf
from tfcalc.service import Service
from tfcalc.model import Model
if __name__ == '__main__':
    service = Service()
    model = Model()
    def print_menu():
        print('0. 종료')
        print('1. +')
        print('2. -')
        print('3. x')
        print('4. /')
        return input('메뉴 입력\n')


    while 1:
        num1 = input('숫자1 입력')
        a = tf.constant(int(num1)) # 변수는 텐서플로우 안에 있다.
        menu = print_menu()
        num2 = input('숫자2 입력')
        b = tf.constant(int(num2))
        model.num1 = a
        model.num2 = b
        if menu == '1':
            print(service.plus(model))
        if menu == '2':
            print(service.minus(model))
        if menu == '3':
            print(service.multiple(model))
        if menu == '3':
            print(service.devide(model))
        elif menu == '0':
            break