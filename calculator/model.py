"""
계산기모델
"""
class Calulator:
    def __init__(self,num1,num2):
        #생성자
        self.num1 = num1
        self.num2 = num2
    def add(self):
        return self.num1 + self.num2
    def sub(self):
        return self.num1 - self.num2
    def multi(self):
        return self.num1 * self.num2
    def divde(self):
        return self.num1 / self.num2