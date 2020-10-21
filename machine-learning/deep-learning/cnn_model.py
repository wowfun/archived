import os
import numpy as np
import tensorflow as tf


# .compat.v1
# import matplotlib.pyplot as plt

class KwArgs:
    def __init__(self):
        self.data_root_path = "E:\Projects\cnn\dataTest"
        self.input_shape=(128,128,3,1)
        # conv 1
        self.conv1_num_filters = 16
        self.conv1_kernel_size = [3, 3, 1]
        self.conv1_strides = (1, 1, 1)

        # pool 1
        self.pool1_size = (2, 2, 1)
        self.pool1_strides = (2, 2, 1)
        # conv 2
        self.conv2_num_filters = 32
        self.conv2_kernel_size = (2, 2, 1)
        self.conv2_strides = (2, 2, 1)

        # pool 2
        self.pool2_size = (2, 2, 1)
        self.pool2_strides = (2, 2, 1)

        # conv 3
        self.conv3_num_filters = 64
        self.conv3_kernel_size = [3, 3, 32]
        self.conv3_strides = (2, 2, 1)

        # pool 3
        self.pool3_size = (2, 2, 1)
        self.pool3_strides = (2, 2, 1)

        # fc
        self.num_fc1_units = 1024
        self.num_fc2_units = 32
        self.num_output_units = 1

        self.dropout_rate = 0.25
        self.batch_size = 32
        self.epochs = 10
        self.learning_rate = 0.0000001
        self.model_save_path = "saved_model"


def read_data(data_dir):
    datas = []
    labels = []
    fpaths = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)  # 获取到矩阵文件夹
        fpaths.append(fpath)

        data = np.load(fpath)
        name = fname.split(".")[0] + '.' + fname.split(".")[1]

        label = float(name.split("_")[-1])

        datas.append(data)
        labels.append(label)
    datas = np.array(datas)  # datas为n*128*128*3*1
    labels = np.array(labels)  # 为时间
    tmp = labels.shape[0]
    labels = labels.reshape(tmp, 1)

    print("shape of datas: {}\tshape of labels: {}".format(datas.shape, labels.shape))
    return datas, labels


def train(datas, labels, kwargs):
    save_model_cb = tf.keras.callbacks.ModelCheckpoint(kwargs.model_save_path, save_weights_only=True, verbose=1,
                                                       save_freq='epoch')
    model = cnn(kwargs)
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mae', 'mse'])
    model.fit(datas, labels, batch_size=kwargs.batch_size, epochs=kwargs.epochs, callbacks=[save_model_cb])
    return model


def cnn(kwargs):
    net = tf.keras.models.Sequential()
    # 第1层卷积
    net.add(tf.keras.layers.Conv3D(
        filters=kwargs.conv1_num_filters,
        kernel_size=kwargs.conv1_kernel_size,
        strides=kwargs.conv1_strides,
        padding='SAME',
        data_format='channels_last',
        activation=tf.nn.relu,
        use_bias=True,
        input_shape=kwargs.input_shape
    ))

    net.add(tf.keras.layers.MaxPool3D(
        pool_size=kwargs.pool1_size,
        strides=kwargs.pool1_strides,
        #    padding='SAME',
    ))

    net.add(tf.keras.layers.Conv3D(
        filters=kwargs.conv2_num_filters,
        kernel_size=kwargs.conv2_kernel_size,
        strides=kwargs.conv2_strides,
        padding='SAME',
        data_format='channels_last',
        activation=tf.nn.relu,
        use_bias=True
    ))

    net.add(tf.keras.layers.MaxPool3D(
        pool_size=kwargs.pool2_size,
        strides=kwargs.pool2_strides,
        padding='SAME',
        data_format='channels_last'
    ))

    net.add(tf.keras.layers.Conv3D(
        filters=kwargs.conv3_num_filters,
        kernel_size=kwargs.conv3_kernel_size,
        strides=kwargs.conv3_strides,
        padding='SAME',
        data_format='channels_last',
        activation=tf.nn.relu,
        use_bias=True
    ))

    net.add(tf.keras.layers.MaxPool3D(
        pool_size=kwargs.pool3_size,
        strides=kwargs.pool3_strides,
        padding='SAME',
        data_format='channels_last'
    ))

    net.add(tf.keras.layers.Flatten())
    net.add(tf.keras.layers.Dense(kwargs.num_fc1_units, activation='relu'))
    net.add(tf.keras.layers.Dense(kwargs.num_fc2_units, activation='relu'))
    net.add(tf.keras.layers.Dense(kwargs.num_output_units, activation='softmax'))

    net.summary()
    return net


def test(model, datas, labels, kwargs):
    test_loss, test_mae, test_mse = model.evaluate(datas, labels, verbose=1)
    print("*** test loss: ", test_loss)
    print("*** test mae: ", test_mae)


def pred(model, data):
    preds = model.predict(data).flatten()
    print("*** pred: ", preds)


if __name__ == '__main__':
    kwargs = KwArgs()
    datas, labels = read_data(kwargs.data_root_path)
    train(datas, labels, kwargs)
