
from oop import Layer_Dense, Activation_ReLU  


X = [
    [1,2,3,2.5],
    [2,5,-1,2],
    [-1.5,2.7,3.3,-0.8]
]
layer1 = Layer_Dense(4,5)
activation1 = Activation_ReLU()
layer1.forward(X)
print("before")
print(layer1.output)
activation1.forward(layer1.output)
print("after")
print(activation1.output)


# X = [
#     [1,2,3,2.5],
#     [2,5,-1,2],
#     [-1.5,2.7,3.3,-0.8]
# ]
# layer1 = Layer_Dense(4, 5)
# layer2 = Layer_Dense(5, 2)
# layer3 = Layer_Dense(2, 1)

# layer1.forward(X)
# layer2.forward(layer1.output)
# layer3.forward(layer2.output)
# print("layer3")
# print(layer3.output)


# X2 = [
#     [1,2,3,2.5],
# ]
# layer1 = Layer_Dense(4, 5)
# layer2 = Layer_Dense(5, 2)
# layer3 = Layer_Dense(2, 1)

# layer1.forward(X2)
# layer2.forward(layer1.output)
# layer3.forward(layer2.output)
# print("layer3")
# print(layer3.output)
