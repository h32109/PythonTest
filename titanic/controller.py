from titanic.service import Service
from titanic.view import View

class Controller:
    def __init__(self):
        self._service = Service()
        self._list = []

    @property
    def list(self) -> object:
        return self._list

    @list.setter
    def list(self, list):
        self._list = list

    def create_model(self, fname) -> object:
        return self._service.new_model(fname)


    def preprocess(self, param) -> object: # 재사용성을 생각해야한다.
        print('----------------1. Cabin Ticket 삭제 -----------------------')
        result = self._service.drop_feature(param, 'Cabin')
        result = self._service.drop_feature(result, 'Ticket')
        print('----------------2. embarked	승선한 항구명 norminal 편집-----------------------')
        result = self._service.embarked_norminal(result)
        print('----------------3. Title 편집 -----------------------')
        result = self._service.title_norminal(result)
        print('----------------4. Name,PassengerId 삭제 -----------------------')
        result = self._service.drop_feature(result, 'Name')
        result = self._service.drop_feature(result, 'PassengerId')
        print('----------------5. Age 편집 -----------------------')
        result = self._service.age_ordinal(result)
        print('----------------6. Fare ordinal 편집 -----------------------')
        result = self._service.fare_ordinal(result)
        print('----------------7. Fare 삭제 -----------------------')
        result = self._service.drop_feature(result, 'Fare')
        print('----------------7. Sex norminal 편집 -----------------------')
        result = self._service.sex_norminal(result)
        result = result.fillna({"FareBand": 1})
        return result

    def create_dummy(self) -> object:
        train = self._train
        dummy = train['Survived']
        return dummy

    def test_all(self):
        model = self.create_model()
        dummy = self.create_dummy()
        m = self._m
        m.hook_test(model, dummy)
