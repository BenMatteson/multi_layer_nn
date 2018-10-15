import sys
import numpy as np
import random
import math
from scipy.special import expit #sigmoid activation function

HIDDEN_LAYER_SIZE = 20
OUTPUT_COUNT = 10
LEARNING_RATE = .01
MOMENTUM = .9

#sigmoid = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))

 # load dataset from csv file as array of tags and matrix of data
def get_dataset_from_file(path):
    with open(path) as dataset_file:
        data_set = []
        tags = []
        input = dataset_file.readline()
        while input != '':
            # get tag value
            tag, pixels = input.split(',', 1)
            # get inputs as lsit
            pixels = pixels.split(',')
            # scale (and convert) values, include bias as first value
            scaled_list = [1] + [float(x) / 255 for x in pixels]
            data_set.append(scaled_list)
            tags.append(int(tag))
            input = dataset_file.readline()
        data_set = np.asarray(data_set)
        #tags = np.asarray(tags)
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
    return np.asarray(matrix)

def guess_from_vector(vector):
    largest_value = float('-inf')
    for index, val in enumerate(vector):
        if val > largest_value:
            guess = index
            largest_value = val
    return guess

def forward_propogate(dataset, hidden_layer_weights, output_layer_weights):
    count = len(dataset[0])
    hidden_layer = np.c_[np.ones(count) , dataset[1] @ hidden_layer_weights]
    #hidden_layer = expit(hidden_layer) # slower, but arguably more correct, does not effect predictions since it preserves reletive order
    output = hidden_layer @ output_layer_weights
    #output = expit(output) # slower, but arguably more correct, does not effect predictions since it preserves reletive order
    return output

# compute the accuracy of the given perceptrons at predicting the given inputs
def compute_accuracy(dataset, hidden_layer_weights, output_layer_weights):
    count = len(dataset[0])
    correct = 0
    output = forward_propogate(dataset, hidden_layer_weights, output_layer_weights)
    for tag, result in zip(dataset[0], output):
        if tag == guess_from_vector(result):
            correct += 1
    return correct / count

# args - training.csv testing.csv
def main(argv=None):
    if argv is None:
        argv = sys.argv
    test_data_path = argv.pop()
    training_data_path = argv.pop()
    training_data = get_dataset_from_file(training_data_path)
    test_data = get_dataset_from_file(test_data_path)
    print('Finished Importing data')
    
    # initialize
    input_size = len(training_data[1][0]) # includes bias unit
    hidden_layer_weights = generate_weight_matrix(input_size, HIDDEN_LAYER_SIZE)
    output_weights = generate_weight_matrix(HIDDEN_LAYER_SIZE + 1, OUTPUT_COUNT) # extra row for bias weights


    # targets for each node indexed by target digit
    targets = (np.ones((10,10)) * .1) + (np.identity(OUTPUT_COUNT) * .8)
    with open('output.txt', 'w') as output:
        output.write('epoch \ttrain \ttest\n')
        output.write('0\t' + str(compute_accuracy(training_data, hidden_layer_weights, output_weights)) + ' \t')
        output.write(str(compute_accuracy(test_data, hidden_layer_weights, output_weights)) + '\n')
        output.flush()
        # initial accuracy
        print('Epoch 0')
        #print(compute_accuracy(training_data, hidden_layer_weights, output_weights))
        #print(compute_accuracy(test_data, hidden_layer_weights, output_weights))

        # begin training
        previous_OWD = 0
        previous_HWD = 0
        for epoch in range(1,50):
            for tag, input in zip(training_data[0], training_data[1]):
                #input = np.asarray(input)
                # activation of hidden nodes
                hidden_nodes = input @ hidden_layer_weights
                hidden_nodes = np.r_[[1], expit(hidden_nodes)] # add bias (and sigmoid)
                # activation of output nodes 
                outputs = hidden_nodes @ output_weights
                outputs = expit(outputs)
                # determine errors
                target = targets[tag,:]
                output_errors = outputs * (1 - outputs) * (target - outputs)
                hidden_errors = hidden_nodes * (1 - hidden_nodes) * (output_weights @ output_errors)
                # update weights for outputs
                output_weight_deltas = np.outer(LEARNING_RATE * output_errors, hidden_nodes).T
                output_weight_deltas = output_weight_deltas + (MOMENTUM * previous_OWD)
                output_weights = output_weights + output_weight_deltas
                previous_OWD = output_weight_deltas
                # update weights for hidden nodes
                hidden_weight_deltas = np.outer((LEARNING_RATE * hidden_errors[1::]), input).T # ignore bias
                hidden_weight_deltas = hidden_weight_deltas + (MOMENTUM * previous_HWD)
                hidden_layer_weights = hidden_layer_weights + hidden_weight_deltas
                previous_HWD = hidden_weight_deltas
            print('Epoch ' + str(epoch))
            #print(compute_accuracy(training_data, hidden_layer_weights, output_weights))
            #print(compute_accuracy(test_data, hidden_layer_weights, output_weights)))
            output.write(str(epoch) + '\t' + str(compute_accuracy(training_data, hidden_layer_weights, output_weights)) + ' \t')
            output.write(str(compute_accuracy(test_data, hidden_layer_weights, output_weights)) + '\n')
            output.flush()
        matrix = [[0 for x in range(10)] for y in range(10)]
        for tag, input in zip(training_data[0], training_data[1]):
            output = forward_propogate(test_data, hidden_layer_weights, output_weights)
            guess = guess_from_vector(output)
            #build y,x so inner list is rows for easy printing
            matrix[tag][guess] += 1
        for row in matrix:
            for num in row:
                output.write(str(num) + ' \t')
            output.write('\n')
    
    print('Done')

if __name__ == "__main__":
    #sys.exit(main())
    main()
