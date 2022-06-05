# -*- coding: UTF-8 -*-

#main文件执行方式：python /home/project/code/main.py
from train.main import start_train
from smp_pred import predict
from class_pred import class_pred
import os
def train_predict():
    # start_train()
    # class_pred()
    predict()
    # os.chdir( os.path.join("..","temp_data"))
    # os.system("zip result.zip -r  result")
    # os.system("mv result.zip %s"%(os.path.join('..','result')))


if __name__ == '__main__':
    train_predict()


