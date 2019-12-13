import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import seaborn as sns

#한글 깨짐을 막기 위한 설정. 한글 폰트 없기 때문에 다운로드를 받아야 함. 지원되는 폰트가 정해져 있음.
rc('font', family = font_manager
   .FontProperties(fname='C:/Users/bit/AppData/Local/Microsoft/Windows/Fonts/H2GTRE.TTF')
   .get_name())

class View:

    def __init__(self):
        pass


    @staticmethod
    def showPlot(list):
        tr = list[0]
        f, ax = plt.subplots(1, 2, figsize=(18, 8))
        tr['Survived'] \
            .value_counts() \
            .plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)  # 0은 가로 ax는 가로축 , shadow는 그림 상
        ax[0].set_title('0.DEAD VS 1.SERVIVED')
        ax[0].set_ylabel('')
        ax[1].set_title('0.DEAD VS 1.SERVIVED')
        sns.countplot('Survived', data=tr, ax=ax[1])
        plt.show()

    @staticmethod
    def plot_sex(list):
        tr = list[0]
        f, ax = plt.subplots(1, 2, figsize=(18, 8))
        tr['Survived'][tr['Sex'] == 'male'] \
            .value_counts() \
            .plot.pie(explode=[0, 04.1],
                      autopct='%1.1f%%',
                      ax=ax[0],
                      shadow=True)
        tr['Survived'][tr['Sex'] == 'female'] \
            .value_counts() \
            .plot.pie(explode=[0, 0.1],
                      autopct='%1.1f%%',
                      ax=ax[1],
                      shadow=True)
        ax[0].set_title('남성의 생존비율 (0:사망자,1:생존자)')
        ax[1].set_title('여성의 생존비율 (0:사망자,1:생존자)')
        plt.show()
