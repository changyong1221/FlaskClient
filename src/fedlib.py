import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

class BaseModel:
    def __init__(self, model):
        self.model = model
        self.history = None
        self.weights = None

    def load_model(self, file_path, weight=False):
        if weight:
            self.weights = np.load(file_path, allow_pickle=True)
            self.model.set_weights(self.weights)
        else:
            self.model = load_model(file_path)

    def compile(self, *args, **kwargs):
        self.model.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        self.history = self.model.fit(*args, **kwargs)

    def save_model(self, save_path, weight=False):
        if weight:
            w = self.model.get_weights()
            np.save(save_path, w)
        else:
            self.model.save(save_path)

    def evaluate(self, *args, **kwargs):
        score = self.model.evaluate(*args, **kwargs)
        loss = score[0]
        accuracy = score[1]
        return loss, accuracy


class FedClient(BaseModel):
    def __init__(self, model=None, ID=0):
        super().__init__(model)
        self.ID = ID

    def setJob(self, jobAdress):
        pass

    def upload(self, weight=False):
        pass


class FedServer(BaseModel):

    def __init__(self, model=None):
        super().__init__(model)
        self.weights = None
        self.models_data = None

    def load_client_weights(self, model_path_list):
        arr = []
        for model_path in model_path_list:
            arr.append(np.load(model_path, allow_pickle=True))
        self.models_data = np.array(arr)

    def load_client_models(self, model_path_list):
        arr = []
        for model_path in model_path_list:
            c_model = load_model(model_path)
            arr.append(c_model.get_weights())
        self.models_data = np.array(arr)

    def fl_average(self):
        # FL average
        self.weights = np.average(self.models_data, axis=0)
        self.model.set_weights(self.weights)

    def setJob(self, jobAdress):
        pass

    def get_modelsID_list_from_block(self):
        pass

if __name__ == '__main__':
    pass