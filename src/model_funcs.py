import os
import time
import shutil
import src.globals as glo
from src.fedlib import *
from src.dataset_funcs import load_client_dataset, load_all_dataset
from keras.models import Sequential
from keras.layers import Dropout, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D

# settings
dataset_path = "datasets/computer_status_dataset.csv"
features = ['cpu_usage', 'memory_usage', 'disk_usage', 'num_tasks',
            'bandwidth', 'mips', 'cpu_freq', 'cpu_cache', 'ram',
            'ram_freq', 'disk', 'pes_num', 'priority']

def print_log(s):
    with open('log.log', 'a+') as f:
        f.write(s + "\n")
        print(s)


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


def save_pic(path, acc, loss, name):
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(acc)), acc, label=f'{name} Accuracy')
    plt.legend(loc='lower right')
    plt.title('{} Accuracy:{:.2}'.format(name, acc[-1]))
    plt.subplot(1, 2, 2)
    plt.plot(range(len(loss)), loss, label='Training Loss')
    plt.legend(loc='upper right')
    plt.title('{} Loss:{:.2}'.format(name, loss[-1]))
    plt.savefig(path)


def train_one_model():
    glo.set_global_var("train_status", "training")
    global_model_path = glo.get_global_var("global_model_path")
    sub_model_path = glo.get_global_var("sub_model_path")
    epoch = 20
    x_train, y_train, x_test, y_test = load_all_dataset(dataset_path, features, test_size=0.5)

    startTime = time.time()
    client_id = glo.get_global_var("client_id")
    model = FedClient(model=create_model(), ID=client_id)
    model.setJob(jobAdress="x3tg83jx0m4jf8chyp5djas4jf9")
    if os.path.exists(global_model_path):
        model.load_model(global_model_path, weight=True)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=x_train,
              y=y_train,
              batch_size=128,
              epochs=epoch,
              verbose=2)
    loss, acc = model.evaluate(x=x_test,
                               y=y_test,
                               batch_size=128)
    model.save_model(sub_model_path, weight=True)
    model.upload()
    print_log(f"Client-ID:{client_id} , loss:{loss} , acc:{acc} , Time:{time.time() - startTime}")
    print("training done.")
    glo.set_global_var("train_status", "todo")


def has_submodel():
    sub_model_path = glo.get_global_var("sub_model_path")
    if os.path.exists(sub_model_path):
        return True
    else:
        return False


def train_models():
    x_train, y_train, x_test, y_test = load_all_dataset(dataset_path, features, test_size=0.5)

    client_num = 5
    x_train_num = len(x_train)
    x_test_num = len(x_test)
    x_train_per = x_train_num // client_num
    x_test_per = x_test_num // client_num
    global_model_path = glo.get_global_var("global_model_path")

    epoch = 10
    for idx in range(client_num):
        startTime = time.time()
        model = FedClient(model=create_model(), ID=idx)
        model.setJob(jobAdress="x3tg83jx0m4jf8chyp5djas4jf9")
        if os.path.exists(global_model_path):
            model.load_model(global_model_path, weight=True)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x=x_train[idx * x_train_per:(idx + 1) * x_train_per],
                  y=y_train[idx * x_train_per:(idx + 1) * x_train_per],
                  batch_size=128,
                  epochs=epoch,
                  verbose=0)
        loss, acc = model.evaluate(x=x_test[idx * x_test_per:(idx + 1) * x_test_per],
                                   y=y_test[idx * x_test_per:(idx + 1) * x_test_per],
                                   batch_size=128)
        model_path = "models/clients/{}.npy".format(idx)
        model.save_model(model_path, weight=True)
        # model.upload()

        print_log(f"Client-ID:{idx} , loss:{loss} , acc:{acc} , Time:{time.time() - startTime}")


# run test dataset for a list of submodels and return test scores
def submodels_test():
    pass


def merge_models_and_test():
    x_train, y_train, x_test, y_test = load_all_dataset(dataset_path, features, test_size=0.5)

    client_id = glo.get_global_var("client_id")
    client_num = glo.get_global_var("merge_clients_num")
    global_model_path = glo.get_global_var("global_model_path")
    x_test_num = len(x_test)
    x_test_per = x_test_num // client_num
    models_path_list = []
    for i in range(client_num):
        print(f'models/downloads/client-{client_id}/{i}.npy')
        models_path_list.append(f'models/downloads/client-{client_id}/{i}.npy')

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
        print_log(f'client({idx})_loss:{loss}, client({idx})_acc:{acc}')

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
    print_log(
        f'global_avg_loss:{np.mean(global_loss_list)}, global_avg_acc:{np.mean(global_acc_list)}')

    global_model.save_model(global_model_path, weight=True)
    global_model_score = np.mean(global_acc_list) * 1000
    retSet = {"clients_scores": client_score_list, "global_score": int(global_model_score)}
    print(retSet)
    return retSet


# update local model to the newest global model
def update_model():
    global_model_path = glo.get_global_var("global_model_path")
    sub_model_path = glo.get_global_var("sub_model_path")
    x_train, y_train, x_test, y_test = load_all_dataset(dataset_path, features, test_size=0.1)

    # get test score of sub_model
    client_id = glo.get_global_var("client_id")
    client_model = FedClient(model=create_model(), ID=client_id)
    client_model.load_model(sub_model_path, weight=True)
    client_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    client_loss, client_acc = client_model.evaluate(x=x_test, y=y_test, batch_size=128)
    sub_model_score = client_acc * 1000
    print_log(f'client({client_id})_loss:{client_loss}, client({client_id})_acc:{client_acc}')

    # get test score of global_model
    global_model = FedServer(model=create_model())
    global_model.load_model(global_model_path, weight=True)
    global_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    global_loss, global_acc = global_model.evaluate(x=x_test, y=y_test, batch_size=128)
    global_model_score = global_acc * 1000

    print(f"sub_model_score: {sub_model_score}")
    print(f"global_model_score: {global_model_score}")
    if global_model_score > sub_model_score:
        print("global model is better.")
        print("updating local model...")
        if os.path.exists(global_model_path):
            shutil.copyfile(global_model_path, sub_model_path)
        print("local model updated.")
    else:
        print("local model is better.")
        os.remove(global_model_path)
        print("global model dropped.")

# if __name__ == '__main__':
#     update_model()
