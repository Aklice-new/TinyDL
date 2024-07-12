import numpy as np
from typing import List
from pathlib import Path
from layer import Layer, Loss,ReLu, Sigmoid, Softmax, Linear, CrossEntropyLoss,Flatten
from net import Net
from PIL import Image

def one_hot(label, num_classes):
    return np.eye(num_classes)[label]

def read_data(data_path:str):
    input, lables = [], []
    for num_class in range(10):
        class_path = data_path.joinpath(str(num_class))
        for img_path in class_path.iterdir():
            img = Image.open(img_path)
            img = np.array(img, dtype=np.float32)
            img = np.expand_dims(img, axis=0)
            label = one_hot(num_class, 10)
            input.append(img)
            lables.append(label)
    # return input, lables
    return (np.array(input), np.array(lables))

def prepare_data(data_path:str):
    train_data_path = Path(data_path).joinpath('training') 
    test_data_path = Path(data_path).joinpath('testing') 
    train_data = read_data(train_data_path)
    test_data = read_data(test_data_path)
    return train_data, test_data

def test(test_data, net):
    input, lables = test_data
    correct = 0
    for i in range(len(input)):
        x, y = input[i:i+1], lables[i:i+1]
        pred = net.test(x)
        # print(f'Predicted: {pred}, Actual: {np.argmax(y)}')
        if np.argmax(y) == pred:
            correct += 1
    print(f'Accuracy: {100 * correct/len(input)} %')

def train(dataset, epochs:int = 1000, batch_size:int = 100, lr:float = 0.001):
    train_data, test_data = dataset
    input, lables = train_data
    net = Net([Flatten(), Linear(784, 256), ReLu(), Linear(256, 10),Softmax()], CrossEntropyLoss())
    # net.check_param()
    for epoch in range(epochs):
        # shullfe data
        idx = np.random.permutation(len(input))
        input, lables = input[idx], lables[idx]
        all_loss = 0
        for i in range(0, len(train_data), batch_size):
            x, y = input[i:i+batch_size], lables[i:i+batch_size]
            loss = net(x, y)
            all_loss += loss * batch_size
            net.backward()
            net.step(lr)
            net.clear_grad()
        if epoch % 100 == 0:
            test(test_data, net)
            print(f"Epoch: {epoch}, Loss: {all_loss/len(input)}")
    # net.check_param()

if __name__ == '__main__':
    dataset = prepare_data('./MNIST/mnist_png/')
    train(dataset)