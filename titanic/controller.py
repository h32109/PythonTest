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


    def create_model2(self) -> object:
        print('----------------1. Cabin Ticket 삭제 -----------------------')
        dframe = self.drop_feature(dframe, 'Cabin')
        dframe = self.drop_feature(dframe, 'Ticket')
        print('----------------2. embarked	승선한 항구명 norminal 편집-----------------------')
        dframe = self.embarked_norminal(dframe)
        print('----------------3. Title 편집 -----------------------')
        dframe = self.title_norminal(dframe)
        print('----------------4. Name,PassengerId 삭제 -----------------------')
        dframe = self.drop_feature(dframe, 'Name')
        # self._test_id = test['PassengerId']
        dframe = self.drop_feature(dframe, 'PassengerId')
        print('----------------5. Age 편집 -----------------------')
        dframe = self.age_ordinal(dframe)
        print('----------------6. Fare ordinal 편집 -----------------------')
        dframe = self.fare_ordinal(dframe)
        print('----------------7. Fare 삭제 -----------------------')
        dframe = self.drop_feature(dframe, 'Fare')
        print('----------------7. Sex norminal 편집 -----------------------')
        dframe = self.sex_norminal(dframe)
        dframe = dframe.fillna({"FareBand": 1})
        return dframe

    def create_dummy(self) -> object:
        train = self._train
        dummy = train['Survived']
        return dummy

    def test_all(self):
        model = self.create_model()
        dummy = self.create_dummy()
        m = self._m
        m.hook_test(model, dummy)
