import pandas as pd
from keras.utils import np_utils
import src.globals as glo


def load_all_dataset(filename, features, test_size=0.1):
    df = pd.read_csv(filename, names=features)
    df.sample(frac=1)
    n_records = len(df)
    data = df.iloc[:, 0:-1]
    labels = df.iloc[:, -1]
    x_train = data.iloc[:int(n_records * (1 - test_size))]
    y_train = labels.iloc[:int(n_records * (1 - test_size))]
    x_test = data.iloc[int(n_records * (1 - test_size)):]
    y_test = labels.iloc[int(n_records * (1 - test_size)):]

    n_features = 12
    n_classes = 20
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    ranges = [100, 100, 100, 20, 2000, 8000, 5, 128, 2000, 4000, 10000, 5]
    for i in range(len(ranges)):
        x_train.iloc[:, i] /= ranges[i]
        x_test.iloc[:, i] /= ranges[i]
    x_train = x_train.values
    x_test = x_test.values
    x_train = x_train.reshape(len(x_train), 1, n_features, 1)
    x_test = x_test.reshape(len(x_test), 1, n_features, 1)
    y_train = np_utils.to_categorical(y_train, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)
    return x_train, y_train, x_test, y_test


def load_client_dataset():
    file_path = "datasets/computer_status_dataset.csv"
    features = ['cpu_usage', 'memory_usage', 'disk_usage', 'num_tasks',
                'bandwidth', 'mips', 'cpu_freq', 'cpu_cache', 'ram',
                'ram_freq', 'disk', 'pes_num', 'priority']
    x_train, y_train, x_test, y_test = load_all_dataset(file_path, features, test_size=0.5)

    client_num = glo.get_global_var("clients_num")
    x_train_num = len(x_train)
    x_test_num = len(x_test)
    x_train_per = x_train_num // client_num
    x_test_per = x_test_num // client_num

    idx = glo.get_global_var("client_id")
    x_train = x_train[idx * x_train_per:(idx + 1) * x_train_per]
    y_train = y_train[idx * x_train_per:(idx + 1) * x_train_per]
    x_test = x_test[idx * x_test_per:(idx + 1) * x_test_per]
    y_test = y_test[idx * x_test_per:(idx + 1) * x_test_per]
    return x_train, y_train, x_test, y_test