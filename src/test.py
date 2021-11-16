import datetime
import socket
import os
import src.globals as glo
from src.fedlib import *
from src.dataset_funcs import load_client_dataset, load_all_dataset
from keras.models import Sequential
from keras.layers import Dropout, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D

# settings
dataset_path = "../datasets/computer_status_dataset.csv"
features = ['cpu_usage', 'memory_usage', 'disk_usage', 'num_tasks',
            'bandwidth', 'mips', 'cpu_freq', 'cpu_cache', 'ram',
            'ram_freq', 'disk', 'pes_num', 'priority']

def create_model():
    n_features = 12
    n_classes = 20
    model = Sequential()
    input_shape = (1, n_features, 1)
    model.add(Conv2D(filters=32, kernel_size=(1, 3), activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(filters=64, kernel_size=(1, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    return model

def train_one_model():
    global_model_path = "../models/global/client-1/global_model.npy"
    sub_model_path = "../models/train/2.npy"
    epoch = 20
    x_train, y_train, x_test, y_test = load_all_dataset(dataset_path, features, test_size=0.5)

    client_id = 1
    model = FedClient(model=create_model(), ID=client_id)
    model.setJob(jobAdress="x3tg83jx0m4jf8chyp5djas4jf9")
    if os.path.exists(global_model_path):
        model.load_model(global_model_path, weight=True)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train,
              y=y_train,
              batch_size=128,
              epochs=epoch,
              verbose=0)
    loss, acc = model.evaluate(x=x_test,
                               y=y_test,
                               batch_size=128)
    model.save_model(sub_model_path, weight=True)
    model.upload()
    print(f"Client-ID:{client_id} , loss:{loss} , acc:{acc}")
    print("training done.")

def merge_models_and_test():
    x_train, y_train, x_test, y_test = load_all_dataset(dataset_path, features, test_size=0.5)

    client_id = 1
    client_num = 5
    global_model_path = "../models/global/client-1/global_model.npy"
    x_test_num = len(x_test)
    x_test_per = x_test_num // client_num
    models_path_list = []
    for i in range(client_num):
        print(f'../models/downloads/client-{client_id}/{i}.npy')
        models_path_list.append(f'../models/downloads/client-{client_id}/{i}.npy')

    # get test scores of submodels
    client_acc_list = []
    client_loss_list = []
    client_score_list = []
    for idx in range(client_num):
        client_model = FedClient(model=create_model(), ID=idx)
        client_model.load_model(models_path_list[i], weight=True)
        client_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        loss, acc = client_model.evaluate(x=x_test[idx * x_test_per:(idx + 1) * x_test_per],
                                          y=y_test[idx * x_test_per:(idx + 1) * x_test_per], batch_size=128)
        client_acc_list.append(acc)
        client_score_list.append(int(acc*1000))
        client_loss_list.append(loss)
        print(f'client({idx})_loss:{loss}, client({idx})_acc:{acc}')

    # merge global model and test
    global_model = FedServer(model=create_model())
    global_model.load_client_weights(models_path_list)
    global_model.fl_average()
    global_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    global_acc_list = []
    global_loss_list = []
    for idx in range(client_num):
        loss, acc = global_model.evaluate(x=x_test[idx * x_test_per:(idx + 1) * x_test_per],
                                          y=y_test[idx * x_test_per:(idx + 1) * x_test_per], batch_size=128)
        global_acc_list.append(acc)
        global_loss_list.append(loss)
    print(
        f'global_avg_loss:{np.mean(global_loss_list)}, global_avg_acc:{np.mean(global_acc_list)}')

    global_model.save_model(global_model_path, weight=True)
    global_model_score = np.mean(global_acc_list) * 1000
    retSet = {"clients_scores": client_score_list, "global_score": int(global_model_score)}
    print(retSet)


def test_one_model():
    x_train, y_train, x_test, y_test = load_all_dataset(dataset_path, features, test_size=0.5)

    sub_model_path = "../models/train/1.npy"
    idx = 1
    client_model = FedClient(model=create_model(), ID=idx)
    client_model.load_model(sub_model_path, weight=True)
    client_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    loss, acc = client_model.evaluate(x=x_test,
                                      y=y_test, batch_size=128)
    print(f'client({idx})_loss:{loss}, client({idx})_acc:{acc}')


def test_federated_model():
    x_train, y_train, x_test, y_test = load_all_dataset(dataset_path, features, test_size=0.5)

    global_model_path = "../models/train/global.npy"
    sub_model_paths = ["../models/train/1.npy",
                       "../models/train/2.npy"
                       ]
    global_model = FedServer(model=create_model())
    global_model.load_client_weights(sub_model_paths)
    global_model.fl_average()

    global_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    loss, acc = global_model.evaluate(x=x_test,
                                      y=y_test, batch_size=128)
    print(f'global_loss:{loss}, global_acc:{acc}')


if __name__ == '__main__':
    # train_one_model()
    # test_one_model()
    # test_federated_model()
    # merge_models_and_test()
    pass
