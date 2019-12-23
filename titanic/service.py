from titanic.model import Model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

"""
['PassengerId' 고객ID, 
'Survived', 생존여부
'Pclass', 승선권 1 = 1st 2 = 2nd 3 = 3rd
'Name', 
'Sex', 
'Age', 
'SibSp',동반한 형제, 자매, 배우자
'Parch', 동반한 부모, 자식
'Ticket', 티켓번호
'Fare', 요금
'Cabin', 객실번호
'Embarked'] 승선한 항구명 C = 쉐부로, Q = 퀸즈타운, S = 사우스햄톤
"""
class Service:
    def __init__(self):
        self._model = Model()

    def new_model(self, param):
        model = self._model
        model.context='./data'
        model.fname=param
        return pd.read_csv(model.context+'/'+model.fname)

    """
    def hook_process(self, train, test) -> object:
        print('----------------1. Cabin Ticket 삭제 -----------------------')
        t = self.drop_feature(train, test, 'Cabin')
        t = self.drop_feature(t[0], t[1], 'Ticket')
        print('----------------2. embarked	승선한 항구명 norminal 편집-----------------------')
        t = self.embarked_norminal(t[0], t[1])
        print('----------------3. Title 편집 -----------------------')
        t = self.title_norminal(t[0], t[1])
        print('----------------4. Name,PassengerId 삭제 -----------------------')
        t = self.drop_feature(t[0], t[1], 'Name')
        self._test_id = test['PassengerId']
        t = self.drop_feature(t[0], t[1], 'PassengerId')
        print('----------------5. Age 편집 -----------------------')
        t = self.age_ordinal(t[0], t[1])
        print('----------------6. Fare ordinal 편집 -----------------------')
        t = self.fare_ordinal(t[0], t[1])
        print('----------------7. Fare 삭제 -----------------------')
        t = self.drop_feature(t[0], t[1], 'Fare')
        print('----------------7. Sex norminal 편집 -----------------------')
        t = self.sex_norminal(t[0], t[1])
        t[1] = t[1].fillna({"FareBand": 1})
        a = self.null_sum(t[1])
        print('널의 수량 {} 개'.format(a))
        self._test = t[1]
        return t[0]
    """
    @staticmethod
    def null_sum(param) -> int:
        return param.isnull().sum()

    @staticmethod
    def drop_feature(train, test, feature) -> []:
        train = train.drop([feature], axis=1)
        test = test.drop([feature], axis=1)
        return [train, test]

    @staticmethod
    def embarked_norminal(param) -> object:
        print('>>>> embarked_norminal')
        # c_city = train[train['Embarked'] == 'C'].shape[0]
        # s_city = train[train['Embarked'] == 'S'].shape[0]
        # q_city = train[train['Embarked'] == 'Q'].shape[0]
        dframe = param.fillna({"Embarked": "S"})
        city_mapping = {"S": 1, "C": 2, "Q": 3}
        dframe['Embarked'] = dframe['Embarked'].map(city_mapping)
        return dframe

    @staticmethod
    def title_norminal(param) -> object:
        dframe = param
        print('>>>> title_norminal')

        for dataset in dframe:
            dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)

        for dataset in dframe:
            dataset['Title'] \
                = dataset['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
            dataset['Title'] \
                = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
            dataset['Title'] \
                = dataset['Title'].replace(['Mile', 'Ms'], 'Miss')

        dframe[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
        # print(train[['Title','Survived']].groupby(['Title'], as_index=False).mean())
        title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Royal': 5, 'Rare': 6, 'Mne': 7}
        for dataset in dframe:
            dataset['Title'] = dataset['Title'].map(title_mapping)
            dataset['Title'] = dataset['Title'].fillna(0)
        return dframe

    @staticmethod
    def sex_norminal(param) -> []:
        dframe = param
        sex_mapping = {'male': 0, 'female': 1}
        for dataset in dframe:
            dataset['Sex'] = dataset['Sex'].map(sex_mapping)

        return param

    @staticmethod # 지도학습.
    def age_ordinal(param) -> object:
        dframe = param
        dframe['Age'] = dframe['Age'].fillna(-0.5) # 가장 많이 탄 나이대로 하는 경향이 있음. fillna 결측값 대체하기
        bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
        labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
# 왜 나이를 붙였는가 분석자료를 만드는 것
        # 인공지능에게 나의 정보를 넘겼을때 '죽음'이라는 단답을 리턴하는 것이 아니라, 차트, *데이터 시각화* 하여 그것에 대한 증명을 보여줘야 한다.
        dframe['AgeGroup'] = pd.cut(dframe['Age'], bins, labels=labels)
        age_title_mapping = {0: 'Unknown', 1: 'Baby', 2: 'Child',
                             3: 'Teenager', 4: 'Student', 5: 'Young Adult', 6: 'Adult', 7: 'Senior'}
        for x in range(len(param['AgeGroup'])):
            if dframe['AgeGroup'][x] == 'Unknown':
                dframe['AgeGroup'][x] = age_title_mapping[dframe['Title'][x]]
	# 코드와 테스트 코드를 두가지 작성하는 것 지도학습의 특징.

        age_mapping = {'Unknown': 0, 'Baby': 1, 'Child': 2,
                       'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}

        dframe['AgeGroup'] = dframe['AgeGroup'].map(age_mapping)
        return dframe

    @staticmethod
    def fare_ordinal(train, test) -> []:
        train['FareBand'] = pd.qcut(train['Fare'], 4, labels={1, 2, 3, 4})
        test['FareBand'] = pd.qcut(test['Fare'], 4, labels={1, 2, 3, 4})
        return [train, test]

    # 검증 알고리즘 작성

    def hook_test(self, model, dummy):
        print('KNN 활용한 검증 정확도 {} %'.format(self.accuracy_by_knn(model, dummy)))
        print('결정트리 활용한 검증 정확도 {} %'.format(self.accuracy_by_dtree(model, dummy)))
        print('랜덤포리스트 활용한 검증 정확도 {} %'.format(self.accuracy_by_rforest(model, dummy)))
        print('나이브베이즈 활용한 검증 정확도 {} %'.format(self.accuracy_by_nb(model, dummy)))
        print('SVM 활용한 검증 정확도 {} %'.format(self.accuracy_by_svm(model, dummy)))

    @staticmethod
    def create_k_fold():
        k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
        return k_fold # test값을 10개로 쪼갠다 / 섞는다 / 한번 사용한 값은 사용하지 않는다.

    @staticmethod
    def create_random_variables(param, X_feature, Y_features) -> []: # 시험 문제를 만든다.
        the_X_feature = X_feature # 디멘션을 줄여서 적은 컬럼만 타켓으로 한다.
        the_Y_feature = Y_features
        train2, test2 = train_test_split(param, test_size=0.3, random_state=0) # random_state 0은 이전에 했던 문제라도 출제한다 0~1 / 0.3은 전체의 30%에서 문제출제
        train_X = train2[the_X_feature]
        train_Y = train2[the_Y_feature]
        test_X = test2[the_X_feature]
        test_Y = test2[the_Y_feature]
        return [train_X, train_Y, test_X, test_Y]

    def accuracy_by_knn(self, model, dummy):
        clf = KNeighborsClassifier(n_neighbors=13)
        scoring = 'accuracy'
        k_fold = self.create_k_fold()
        score = cross_val_score(clf, model, dummy, cv=k_fold, n_jobs=1, scoring=scoring)
        accuracy = round(np.mean(score) * 100, 2)
        return accuracy

    def accuracy_by_dtree(self, model, dummy): # accuracy return은 string 값이다.
        clf = DecisionTreeClassifier() # 검증타겟
        scoring = 'accuracy'
        k_fold = self.create_k_fold()
        score = cross_val_score(clf, model, dummy, cv=k_fold, n_jobs=1, scoring=scoring) # 프로퍼티에 kfold를 넣어라/ 데이터세트 1개 / accuracy는 정확도
        accuracy = round(np.mean(score) * 100, 2)
        return accuracy

    def accuracy_by_rforest(self, model, dummy):
        clf = RandomForestClassifier(n_estimators=13)  # n_이 나오면 무조건 숫자 estimator = 측정기 수(dtree 수)
        scoring = 'accuracy'
        k_fold = self.create_k_fold()
        score = cross_val_score(clf, model, dummy, cv=k_fold, n_jobs=1, scoring=scoring)
        accuracy = round(np.mean(score) * 100, 2)
        return accuracy

    def accuracy_by_nb(self, model, dummy):
        clf = GaussianNB()
        scoring = 'accuracy'
        k_fold = self.create_k_fold()
        score = cross_val_score(clf, model, dummy, cv=k_fold, n_jobs=1, scoring=scoring)
        accuracy = round(np.mean(score) * 100, 2)
        return accuracy

    def accuracy_by_svm(self, model, dummy):
        clf = SVC()
        scoring = 'accuracy'
        k_fold = self.create_k_fold()
        score = cross_val_score(clf, model, dummy, cv=k_fold, n_jobs=1, scoring=scoring)
        accuracy = round(np.mean(score) * 100, 2)
        return accuracy

    """"@staticmethod
    def create_model_dummy(train): # 모델(프로그램) = 머신
        model = train.drop('Survived', axis = 1) # axis = 1 세로축 / 답만 모르는 부분
        dummy = train['Survived'] # 답만 알고 있는 부분
        return [model, dummy]"""

