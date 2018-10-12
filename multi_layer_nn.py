import sys
import numpy as np
import random
import math

HIDDEN_LAYER_SIZE = 20
OUTPUT_COUNT = 10

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

def main(argv=None):
    if argv is None:
        argv = sys.argv
    training_data_path = argv.pop()
    test_data_path = argv.pop()
    training_data = get_dataset_from_file(training_data_path)
    #test_data = get_dataset_from_file(test_data_path)
    print('Finished Importing data')
    
    # initialize
    input_size = len(training_data[1][0])
    #input_size = training_data[1].shape[1]
    hidden_layer_weights = []
    for input in range(0, input_size):
        hidden_layer_weights.append([])
        for node in range(0, HIDDEN_LAYER_SIZE):
            hidden_layer_weights[input].append(random.uniform(-.05, .05))
    hidden_layer_weights = np.asmatrix(hidden_layer_weights)

    output_weights = []
    for hidden_node in range(0, HIDDEN_LAYER_SIZE):
        output_weights.append([])
        for output in range(0, OUTPUT_COUNT):
            output_weights[hidden_node].append(random.uniform(-.05, .05))
    output_weights = np.asmatrix(output_weights)

    sigmoid = np.vectorize(lambda x: 1 / (1 + (math.e ** x)))
    targets = np.identity(OUTPUT_COUNT) * .9

    # begin training
    for epoch in range(0,1):
        for tag, input in zip(training_data[0], training_data[1]):
            hidden_nodes = input * hidden_layer_weights
            sigmoid(hidden_nodes)

            outputs = hidden_nodes * output_weights
            sigmoid(outputs)
            print(outputs)


    #hidden_layer = training_data[1] * hidden_layer_weights
    #sigmoid(hidden_layer)

    #output = hidden_layer * output_weights
    #sigmoid(output)

    print('Done')

if __name__ == "__main__":
    #sys.exit(main())
    main()
