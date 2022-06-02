# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
######################## train lenet example ########################
train lenet and get network model files(.ckpt) :
python train.py --data_path /YourDataPath
"""

import os
import ast
import argparse

from Tools.scripts.var_access_benchmark import C

from src.config import mnist_cfg as cfg
from src.dataset import create_dataset
from src.lenet import LeNet5
import mindspore.nn as nn
from mindspore import context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train import Model
from mindspore.nn.metrics import Accuracy
from mindspore.common import set_seed


set_seed(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore Lenet Example')
    # 设备设置
    parser.add_argument('--device_target', type=str, default=" Ascend",choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented(default: Ascend)')
    parser.add_argument('--data_path', type=str, default="./MNIST_Data",
                        help='path where the dataset is saved')
    parser.add_argument('--ckpt_path', type=str, default="./ckpt", help='if is test, must provide\
                        path where the trained ckpt file')
    parser.add_argument('--dataset_sink_mode', type=ast.literal_eval, default=True,
                        help='dataset_sink_mode is False or True')

    args = parser.parse_args()


    #context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    ds_train = create_dataset(os.path.join(args.data_path, "train"),
                              cfg.batch_size)
    ds_eval = create_dataset(os.path.join(args.data_path, "test"),
                             cfg.batch_size, 1)


    # 根据数据集存储地址，生成数据集
    def create_dataset(data_dir, training=True, batch_size=32, resize=(32, 32),
                       rescale=1 / (255 * 0.3081), shift=-0.1307 / 0.3081, buffer_size=64, ms=None, CV=None):
        # 生成训练集和测试集的路径
        data_train = os.path.join(data_dir, 'train')  # train set
        data_test = os.path.join(data_dir, 'test')  # test set
        # 利用 MnistDataset 方法读取 mnist 数据集，如果 training 是 True 则读取训练集
        ds = ms.dataset.MnistDataset(data_train if training else data_test)
        # map 方法是非常有效的方法，可以整体对数据集进行处理，resize 改变数据形状，rescale 进行归一化，HWC2CHW改变图像通道
        ds = ds.map(input_columns=["image"], operations=[CV.Resize(resize), CV.Rescale(rescale,shift), CV.HWC2CHW()])
        # 利用 map 方法改变数据集标签的数据类型
        ds = ds.map(input_columns=["label"], operations=C.TypeCast(ms.int32))
        # shuffle 是打乱操作，同时设定了 batchsize 的大小，并将最后不足一个 batch 的数据抛弃
        ds = ds.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)
        return ds
    network = LeNet5(cfg.num_classes)
    #设定loss函数
    """your code here"""
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    #设定优化器
    """your code here"""
    net_opt = nn.Momentum(network.trainable_params(), cfg.lr, cfg.momentum)
    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    config_ck = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_steps,
                                 keep_checkpoint_max=cfg.keep_checkpoint_max)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet", directory=args.ckpt_path, config=config_ck)
    #编译形成模型
    """your code here"""
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    print("============== Starting Training ==============")
    # 训练网络 train.py 
    """your code here"""
    model.train(cfg['epoch_size'], ds_train, callbacks=[time_cb, ckpoint_cb, LossMonitor()],
                dataset_sink_mode=args.dataset_sink_mode)
