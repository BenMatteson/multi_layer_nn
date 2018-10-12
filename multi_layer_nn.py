import sys
import numpy as np
import random
import math

HIDDEN_LAYER_SIZE = 20
OUTPUT_COUNT = 10
LEARNING_RATE = .01
MOMENTUM = .9

def get_dataset_from_file(path):
    with open(path) as dataset_file:
        data_set = []
        tags = []
        input = dataset_file.readline()
        while input != '':
            # get tag value
            tag, pixels = input.split(',', 1)
            # get inputs as lsit
            pix_list = pixels.split(',')
            # scale (and convert) values, include bias as first value
            scaled_list = [1] + [float(x) / 255 for x in pix_list]
            data_set.append(scaled_list)
            tags.append(int(tag))
            input = dataset_file.readline()
        #data_set = np.asmatrix(data_set)
        tagged_set = [tags, data_set]
    return tagged_set

# compute the accuracy of the given perceptrons at predicting the given inputs
def compute_accuracy(dataset, perceptrons):
    count = correct = 0
    for img in dataset:
        if img[0] == guess_didgit(perceptrons, img):
            correct += 1
        count += 1
    return correct / count

# generate random weights for a step in the NN, height is inputs, width is nodes
def generate_weight_matrix(height, width):
    matrix = []
    for row in range(0, height):
        matrix.append([])
        for col in range(0, width):
            matrix[row].append(random.uniform(-.05, .05))
    return np.asmatrix(matrix)

def main(argv=None):
    if argv is None:
        argv = sys.argv
    training_data_path = argv.pop()
    test_data_path = argv.pop()
    training_data = get_dataset_from_file(training_data_path)
    ######################test_data = get_dataset_from_file(test_data_path)
    print('Finished Importing data')
    
    # initialize
    input_size = len(training_data[1][0])
    hidden_layer_weights = generate_weight_matrix(input_size, HIDDEN_LAYER_SIZE)
    output_weights = generate_weight_matrix(HIDDEN_LAYER_SIZE, OUTPUT_COUNT)

    def sigmoid(x): 
        return 1 / (1 + np.exp(-x))
    vsigmoid = np.vectorize(sigmoid)

    targets = (np.ones((10,10)) * .1) + (np.identity(OUTPUT_COUNT) * .8)

    # begin training
    previous_OWD = 0
    previous_HWD = 0
    for epoch in range(0,50):
        for tag, input in zip(training_data[0], training_data[1]):
            hidden_nodes = input * hidden_layer_weights
            vsigmoid(hidden_nodes)

            outputs = hidden_nodes * output_weights
            vsigmoid(outputs)

            target = targets[tag,:]
            output_errors = np.multiply(np.multiply(outputs, (1 - outputs)), np.subtract(target, outputs))
            hidden_errors = np.multiply(np.multiply(hidden_nodes, (1 - hidden_nodes)), np.sum((output_weights * output_errors.T), axis=0))
            
            output_weight_deltas = ((LEARNING_RATE * output_errors).T * hidden_nodes).T + (MOMENTUM * previous_OWD)
            previous_OWD = output_weight_deltas
            output_weights = output_weights + output_weight_deltas
            
            hidden_weight_deltas = ((LEARNING_RATE * hidden_errors).T * input).T + (MOMENTUM * previous_HWD)
            previous_HWD = hidden_weight_deltas
            hidden_layer_weights = hidden_layer_weights + hidden_weight_deltas
        print('Epoch ' + str(epoch))
            


    #hidden_layer = training_data[1] * hidden_layer_weights
    #sigmoid(hidden_layer)

    #output = hidden_layer * output_weights
    #sigmoid(output)

    print('Done')

if __name__ == "__main__":
    #sys.exit(main())
    main()
