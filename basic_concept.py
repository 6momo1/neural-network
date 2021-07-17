import numpy as np

# cols are # of neurons ,rows are the batches

inputs = [
    [1,2,3,2.5],
    # [2,5,-1,2],
    # [-1.5,2.7,3.3,-0.8]
]


# input neurons, 3 nodes
weights = [
    [0.2,0.8,-0.5,1.0],
    [0.5,-0.91,0.26,-0.5],
    [-0.26,-27,0.17,0.87]
]

"""
4 x 3
*   *
*   *
*   *
*

"""

biases = [2,3,0.5]
output = np.dot(inputs, np.array(weights).T) + biases
print(output)
