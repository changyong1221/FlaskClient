import random
import numpy as np
import gzip
import os
import torch


class DataSet(object):
    def __init__(self, clients_num, is_iid):
        # whole data
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None

        self._index_in_train_epoch = 0
        self.clients_num = clients_num
        self.is_iid = is_iid
        self.mnist_dataset_construct()

    def mnist_dataset_construct(self):
        data_dir = 'datasets'
        train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
        train_images = extract_images(train_images_path)
        train_labels = extract_labels(train_labels_path)
        test_images = extract_images(test_images_path)
        test_labels = extract_labels(test_labels_path)

        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        assert train_images.shape[3] == 1
        assert test_images.shape[3] == 1
        train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2])
        test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2])

        train_images = train_images.astype(np.float32)
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        if self.is_iid:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = torch.tensor(train_images[order])
            self.train_label = torch.argmax(torch.tensor(train_labels[order]), dim=1)
        else:
            labels = np.argmax(train_labels, axis=1)
            order = np.argsort(labels)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]

        self.test_data = torch.tensor(test_images)
        self.test_label = torch.argmax(torch.tensor(test_labels), dim=1)

    def get_test_dataset(self):
        return self.test_data, self.test_label

    def get_train_dataset(self, client_id):
        batches_size = (int)(len(self.train_data) / self.clients_num)
        return self.train_data[batches_size*(client_id - 1):batches_size*client_id], \
               self.train_label[batches_size*(client_id - 1):batches_size*client_id]

    def get_train_batch(self, client_id, train_batch_size):
        if self.is_iid:
            batches_size = (int)(self.train_data_size / self.clients_num)
            client_train_data = self.train_data[batches_size*(client_id - 1):batches_size*client_id]
            client_test_data = self.train_label[batches_size*(client_id - 1):batches_size*client_id]
            client_test_data = torch.reshape(client_test_data, (-1, 1))
            client_dataset = torch.concat([client_train_data, client_test_data], dim=1)
            client_dataset_slice = random.sample(client_dataset.tolist(), train_batch_size)
            client_dataset_slice = np.array(client_dataset_slice)
            x_train, y_train = np.hsplit(client_dataset_slice, [784, ])
            x_train = x_train.astype(np.float32)
            y_train = y_train.flatten()
            y_train = y_train.astype(np.int64)
            x_train = torch.tensor(x_train)
            y_train = torch.tensor(y_train)
            return x_train, y_train
        else:
            client_slice_num = 2
            shard_size = self.train_data_size // self.clients_num // client_slice_num
            shards_id = np.random.RandomState(seed=self.clients_num).permutation(self.train_data_size // shard_size)
            shards_id1 = shards_id[(client_id - 1) * 2]
            shards_id2 = shards_id[(client_id - 1) * 2 + 1]
            data_shards1 = self.train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            data_shards2 = self.train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            label_shards1 = self.train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            label_shards2 = self.train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            client_train_data = local_data
            client_test_data = np.argmax(local_label, axis=1)
            client_test_data = np.reshape(client_test_data, (-1, 1))
            client_dataset = np.concatenate([client_train_data, client_test_data], axis=1)
            client_dataset_slice = random.sample(client_dataset.tolist(), train_batch_size)
            client_dataset_slice = np.array(client_dataset_slice)
            x_train, y_train = np.hsplit(client_dataset_slice, [784, ])
            x_train = x_train.astype(np.float32)
            y_train = y_train.flatten()
            y_train = y_train.astype(np.int64)
            x_train = torch.tensor(x_train)
            y_train = torch.tensor(y_train)
            return x_train, y_train


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    # print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    # print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return dense_to_one_hot(labels)


if __name__ == "__main__":
    client_dataset = DataSet(10, True)
    train_data, train_labels = client_dataset.get_train_batch(1, 64*10)