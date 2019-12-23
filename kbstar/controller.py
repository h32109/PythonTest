from kbstar.service import Service
class Controller:
    def __init__(self):
        self._service = Service()

    def create_model(self, fname):
        return self._service.new_model(fname)