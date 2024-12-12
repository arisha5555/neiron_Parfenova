import numpy as np
import cv2
from PIL import ImageTk, Image, ImageDraw, ImageOps


# активационные функции и их производные
def sigmoid(x):
    '''
    activation function

    :param x: Param function
    :type x: float
    :return: The sigmoid function for the parameter x
    :rtype: float
    '''
    return 1 / (1 + np.exp(-x))


def proizvodnaya_sigmoid(x):
    '''
    the derivative of the activation function

    :param x: Param function
    :type x: float
    :return: The derivative of the sigmoid function for the parameter X
    :rtype: float
    '''
    return x * (1 - x)


def vivod(output):
    '''
    output of the neural network result

    :param output: result
    :type output: float
    :return: the result on the screen
    '''
    if output < 0.1:
        print('-')
    else:
        print('+')
    print("Output:", output)


def image_to_vector(image_path):
    '''
    converts image to a vector

    :param image_path: The path to the image on the disk
    :type image_path: str
    :return: array of image points
    :rtype: array
    '''

    # Загрузка изображения
    image = cv2.imread(image_path)

    # Преобразование изображения в оттенки серого (необязательно)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Преобразование изображения в матрицу
    matrix = np.array(gray_image)

    image_to_vector = []
    for i in matrix:
        for j in i:
            image_to_vector.append(sigmoid(int(j)))

    return image_to_vector


# класс Neuron
class Neuron:
    '''
    class initialization start value
    '''

    def __init__(self, input_size, activation='sigmoid'):
        '''
        :param input_size: the number of pixels in the image
        :type input_size: int
        :param activation: the name of the neuron activation function
        :type activation: str
        '''
        self.weights = np.random.randn(input_size + 1) * 0.1
        self.activation_function = self._get_activation_function(activation)
        self.proizvodnaya_sigmoid = self._get_activation_proizvodnaya(activation)
        self.output = None

    def _get_activation_function(self, activation):
        '''
        :param activation: the name of the neuron activation function
        :type activation: str
        :return: the sigmoid function
        '''
        if activation == 'sigmoid':
            return sigmoid

    def _get_activation_proizvodnaya(self, activation):
        '''
        :param activation: the name of the neuron activation function
        :type activation: str
        :return: the derivative sigmoid function
        '''
        if activation == 'sigmoid':
            return proizvodnaya_sigmoid

    def activate(self, inputs):
        '''
        :param inputs: incoming array of real numbers
        :type inputs: float
        :return: array of real numbers
        :rtype: float array
        '''
        # умножает входной слой на веса и добавляет смещение
        z = np.dot(inputs, self.weights[:-1]) + self.weights[-1]
        self.output = self.activation_function(z)  # применяем сигмоиду
        return self.output


# класс Layer = слой
class Layer:
    '''
    network layer
    '''

    def __init__(self, num_neurons, input_size, activation='sigmoid'):
        '''
        :param num_neurons: the number of neurons in the hidden layer
        :type num_neurons: int
        :param input_size: the number of pixels in the image
        :type input_size: int
        :param activation: the name of the neuron activation function
        :type activation: str
        '''
        self.neurons = [Neuron(input_size, activation) for _ in range(num_neurons)]
        self.output = None
        self.error = None
        self.delta = None

    def forward(self, inputs):
        '''
        calculation of the activation function for the incoming array

        :param inputs: incoming array of real numbers
        :type inputs: float
        :return: array with a calculated activation function
        :rtype: float array
        '''
        # для входящего вектора изображения расчитывает фукцию activate
        self.output = np.array([neuron.activate(inputs) for neuron in self.neurons])
        return self.output


# класс Network = сеть
class Network:
    '''
    class for calculating neiroset
    '''

    def __init__(self, layers):
        '''
        :param layers: array of layer values
        :type layers: float
        '''
        self.layers = layers

    def forward(self, X):
        '''
        :param X: input values
        :type X: array
        :return: values of the layer elements
        '''
        for layer in self.layers:
            # применяем прямой счет, расчитываем h
            X = layer.forward(X)
        return X

    def backward(self, X, y, learning_rate):
        '''
        :param X: array of images
        :type X: array
        :param y: array of true values
        :type y: array
        :param learning_rate: a characteristic showing the rate of updating the weights
        :type learning_rate: float
        :return: refined weights
        :rtype: float
        '''
        # прямое распространение для получения выходных значений
        output = self.forward(X)

        # вычисление ошибки на выходном слое
        # посчитали ошибку = то, что должны были получить - то, что получили
        error = y - output
        self.layers[-1].error = error
        # градиент на выходном слое
        self.layers[-1].delta = error * self.layers[-1].neurons[0].proizvodnaya_sigmoid(output)

        # передача ошибки обратно через слои
        for i in reversed(range(len(self.layers) - 1)):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]
            # расчитывает ошибку для всех слоев
            layer.error = np.dot(next_layer.delta, np.array([neuron.weights[:-1] for neuron in next_layer.neurons]))
            # вычисляет градиент для всех слоев
            layer.delta = layer.error * np.array(
                [neuron.proizvodnaya_sigmoid(neuron.output) for neuron in layer.neurons])
        # обновление весов
        for i in range(len(self.layers)):
            layer = self.layers[i]
            inputs = X if i == 0 else self.layers[i - 1].output
            for j, neuron in enumerate(layer.neurons):
                for k in range(len(neuron.weights) - 1):
                    neuron.weights[k] += learning_rate * layer.delta[j] * inputs[k]
                neuron.weights[-1] += learning_rate * layer.delta[j]  # Обновление смещения

    def train(self, X, y, learning_rate, epochs):
        '''
        :param X: array of images
        :type X: array
        :param y: array of true values
        :type y: array
        :param learning_rate: a characteristic showing the rate of updating the weights
        :type learning_rate: float
        :param epochs: the number of iterations in the learning process
        :type epochs: int
        '''
        for epoch in range(epochs):
            # zip делает кортежи из вектора изображения и значения, которое мы должны получить на выходе
            for xi, yi in zip(X, y):
                # обратный счет
                self.backward(xi, yi, learning_rate)


# создание и обучение сети
input_size = 400
'''
the number of pixels in the image
'''
hidden_size = 25
'''
the number of neirons in the hidden layer
'''
output_size = 1
'''
the number of neurons in the output layer
'''

layer1 = Layer(hidden_size, input_size, activation='sigmoid')
'''
installing the parameters layer1(hidden)
'''
layer2 = Layer(output_size, hidden_size, activation='sigmoid')
'''
installing the parameters layer2(output)
'''
network = Network([layer1, layer2])
'''
installing the parameters neural network
'''

# пример данных для тренировки
image_plus_1 = image_to_vector("C:\project\plus_1.jpg")
'''
image for neural network training
'''
image_plus_2 = image_to_vector("C:\project\plus_2.jpg")
'''
image for neural network training
'''
image_plus_3 = image_to_vector("C:\project\plus_3.jpg")
'''
image for neural network training
'''
image_plus_4 = image_to_vector("C:\project\plus_4.jpg")
'''
image for neural network training
'''
image_plus_5 = image_to_vector("C:\project\plus_5.jpg")
'''
image for neural network training
'''
image_plus_6 = image_to_vector("C:\project\plus_6.jpg")
'''
image for neural network training
'''
image_plus_7 = image_to_vector("C:\project\plus_7.jpg")
'''
image for neural network training
'''
image_plus_8 = image_to_vector("C:\project\plus_8.jpg")
'''
image for neural network training
'''
image_minus_1 = image_to_vector("C:\project\minus_1.jpg")
'''
image for neural network training
'''
image_minus_2 = image_to_vector("C:\project\minus_2.jpg")
'''
image for neural network training
'''
image_minus_3 = image_to_vector("C:\project\minus_3.jpg")
'''
image for neural network training
'''
image_minus_4 = image_to_vector("C:\project\minus_4.jpg")
'''
image for neural network training
'''
image_minus_5 = image_to_vector("C:\project\minus_5.jpg")
'''
image for neural network training
'''
image_minus_6 = image_to_vector("C:\project\minus_6.jpg")
'''
image for neural network training
'''
image_minus_7 = image_to_vector("C:\project\minus_7.jpg")
'''
image for neural network training
'''
image_minus_8 = image_to_vector("C:\project\minus_8.jpg")
'''
image for neural network training
'''

X = np.array((image_plus_1, image_plus_2, image_plus_3, image_plus_4, image_plus_5, image_plus_6, image_plus_7,
              image_plus_8, image_minus_1, image_minus_2, image_minus_3, image_minus_4, image_minus_5, image_minus_6,
              image_minus_7, image_minus_8))
'''
array of images
'''
y = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [0], [0], [0], [0], [0], [0], [0], [0]])
'''
array of true values
'''

# параметры обучения
learning_rate = 0.4  # скорость обучения
'''
a characteristic showing the rate of updating the weights
'''
epochs = 1000
'''
the number of iterations in the learning process
'''

# обучение сети
network.train(X, y, learning_rate, epochs)

# тестирование сети

image_plus_proverka1 = image_to_vector("C:\project\plus_pohog.jpg")
'''
image for checking the performance of a neural network
'''
X = np.array((image_plus_proverka1))
output = network.forward(X)
'''
the result on the screen
'''
vivod(output)
#
# image_plus_proverka2 = image_to_vector("C:\project\plus_1.jpg")
# '''
# image for checking the performance of a neural network
# '''
# X = np.array((image_plus_proverka2))
# output = network.forward(X)
# '''
# the result on the screen
# '''
# vivod(output)
#
# image_minus_proverka1 = image_to_vector("C:\project\minus_3.jpg")
# '''
# image for checking the performance of a neural network
# '''
# X = np.array((image_minus_proverka1))
# output = network.forward(X)
# '''
# the result on the screen
# '''
# vivod(output)