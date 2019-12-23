import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import seaborn as sns
from titanic.model import Model

#한글 깨짐을 막기 위한 설정. 한글 폰트 없기 때문에 다운로드를 받아야 함. 지원되는 폰트가 정해져 있음.
rc('font', family =
   font_manager
   .FontProperties(fname='C:/Users/bit/AppData/Local/Microsoft/Windows/Fonts/H2GTRE.ttf')
   .get_name())

class View:
    def __init__(self, model):
        self._model = model

    def plot_survived_dead(self):
        model = self._model
        f, ax = plt.subplots(1, 2, figsize=(18, 8))
        model['Survived'] \
            .value_counts() \
            .plot.pie(explode=[0, 0.1],
                      autopct='%1.1f%%',
                      ax=ax[0],
                      shadow=True) # 0은 가로 ax는 가로축 , shadow는 그림 상
        ax[0].set_title('0.사망자 VS 1.생존자')
        ax[0].set_ylabel('')
        ax[1].set_title('0.사망자 VS 1.생존자')
        sns.countplot('Survived', data=model, ax=ax[1])
        plt.show()

    def plot_sex(self):
        model = self._model
        f, ax = plt.subplots(1, 2, figsize=(18, 8))
        model['Survived'][model['Sex'] == 'male'] \
            .value_counts() \
            .plot.pie(explode=[0, 0.1],
                      autopct='%1.1f%%',
                      ax=ax[0],
                      shadow=True)
        model['Survived'][model['Sex'] == 'female'] \
            .value_counts() \
            .plot.pie(explode=[0, 0.1],
                      autopct='%1.1f%%',
                      ax=ax[1],
                      shadow=True)
        ax[0].set_title('남성의 생존비율 (0:사망자,1:생존자)')
        ax[1].set_title('여성의 생존비율 (0:사망자,1:생존자)')
        plt.show()