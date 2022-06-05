
from .smp_train import train
from .class_train import train_class
def start_train():
    # train_class()
    all_models = ['unpp_ns']
    for model in all_models:
        train(model=model)
if __name__ == '__main__':
    start_train()

