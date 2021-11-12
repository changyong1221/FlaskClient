from src.fedlib import *
import pandas as pd
from keras.models import Sequential
from keras.layers import Dropout, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils


def print_log(s):
	with open('log.log', 'a+') as f:
		f.write(s + "\n")
	print(s)


def load_dataset(filename, features, test_size=0.1):
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


if __name__ == '__main__':
	import time

	file_path = "../datasets/computer_status_dataset.csv"
	features = ['cpu_usage', 'memory_usage', 'disk_usage', 'num_tasks',
	            'bandwidth', 'mips', 'cpu_freq', 'cpu_cache', 'ram',
	            'ram_freq', 'disk', 'pes_num', 'priority']
	x_train, y_train, x_test, y_test = load_dataset(file_path, features, test_size=0.5)

	client_num = 10
	x_train_num = len(x_train)
	x_test_num = len(x_test)
	x_train_per = x_train_num // client_num
	x_test_per = x_test_num // client_num

	global_model = FedServer(model=create_model())
	global_model_save_path = "GlobalModels/global.mods.npy"
	global_model.save_model(global_model_save_path, weight=True)
	loss_list = []
	acc_list = []
	global_epoch = 200

	for epoch in range(global_epoch):
		local_loss_list = []
		local_acc_list = []
		models_path_list = []

		for idx in range(client_num):
			startTime = time.time()
			model = FedClient(model=create_model(), ID=idx)
			model.setJob(jobAdress="x3tg83jx0m4jf8chyp5djas4jf9")
			model.load_model(global_model_save_path, weight=True)
			model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
			model.fit(x=x_train[idx * x_train_per:(idx + 1) * x_train_per],
			          y=y_train[idx * x_train_per:(idx + 1) * x_train_per],
			          batch_size=128,
			          epochs=200,
			          verbose=0)
			loss, acc = model.evaluate(x=x_test[idx * x_test_per:(idx + 1) * x_test_per],
			                           y=y_test[idx * x_test_per:(idx + 1) * x_test_per],
			                           batch_size=128)
			model_path = "ClientModels/{}.npy".format(idx)
			model.save_model(model_path, weight=True)
			model.upload()

			print_log(f"Client-ID:{idx} , loss:{loss} , acc:{acc} , Time:{time.time() - startTime}")

		print_log(f'epoch:{epoch} , avg_loss:{np.mean(local_loss_list)}, avg_acc:{np.mean(local_acc_list)}')
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
			f'epoch:{epoch} , global_avg_loss:{np.mean(global_loss_list)}, global_avg_acc:{np.mean(global_acc_list)}')
		loss_list.append(np.mean(global_loss_list))
		acc_list.append(np.mean(global_acc_list))
		global_model.save_model(global_model_save_path, weight=True)
	save_pic(path="GlobalModels/global.jpg", acc=acc_list, loss=loss_list, name="Global_acc_loss.jpg")
