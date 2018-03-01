# dataset: http://archive.ics.uci.edu/ml/datasets/semeion+handwritten+digit
# inspiration: https://iamtrask.github.io/2015/07/12/basic-python-network/
#
import numpy
import os
import random
from terminaltables import AsciiTable


# logistic function
def sigmoid(x):
    return 1.0 / (1.0 + numpy.exp(-x))


# derivative of logistic function
def dsigmoid(x):
    return x * (1.0 - x)


# computes result from [1x256] sample, requires first_layer and second_layer to be defined globally
# returns single detected number
def compute_result(input_sample):
    # process input vector through both layers on NN
    l1 = sigmoid(numpy.dot(input_sample, first_layer))
    l2 = sigmoid(numpy.dot(l1, second_layer))

    # loop through all numbers in sequence and return index of highest value
    maximum = 0
    selected_index = 0
    for index in xrange(10):
        if l2[index] > maximum:
            maximum = l2[index]
            selected_index = index

    return selected_index


# converts [1x256] sample line into pretty 16x16 character block where 1 is * and other symbols are omitted
def print_sample(input_sample):
    # convert [1x256] matrix to [16x16]
    input_sample = input_sample.reshape(16, 16).tolist()

    text = []

    # process sample row by row
    for sample_row in xrange(16):
        text_row = input_sample[sample_row]
        # replace 1 with * and 0 with empty space
        text_row = map(lambda cell: '*' if cell == 1 else ' ', text_row)
        # join 16 characters into line
        text_row = ''.join(text_row)
        # line to rows array
        text.append(text_row)

    # finally, join rows with newlines
    return '\n'.join(text)


# debug settings for outputting
numpy.set_printoptions(threshold='nan', suppress=True)

# we need to tell numpy the dimensions of our arrays
samples = numpy.empty([0, 256])
results = numpy.empty([0, 10])

with open(os.path.dirname(os.path.realpath(__file__)) + '/semeion.data') as file:
    for line in file:
        # split line to array using space as separator
        numbers = line.split(' ')
        # as line read from the file is always is string, we need to convert first 256 parts to decimals,
        # and following 10 to integers
        sample = list(map(lambda x: float(x), numbers[0:256]))
        result = list(map(lambda x: int(x), numbers[256:266]))

        # after that, append freshly read sample and result to arrays
        sample = numpy.array([sample])
        result = numpy.array([result])

        samples = numpy.concatenate((samples, sample), axis=0)
        results = numpy.concatenate((results, result), axis=0)

# numpy.random returns 0..1, by multiplying by 2 we get 0..2,
# by subtracting 1 we get -1..1, and by division by 100 we get -0.01..0.01
first_layer = (2 * numpy.random.random((256, 256)) - 1) / 100  # the array has 256x256 dimensions
second_layer = (2 * numpy.random.random((256, 10)) - 1) / 100  # the array has 256x10 dimensions

# rate defines how fast out network will change. Smaller values leads to slower but more precise training
rate = 0.4

# initial value of error must be high
error = 1000.0
# current epoch
epoch = 1
# limit of epochs
epoch_limit = 50
# we stop after error is that small
desired_error = 0.1

while epoch < epoch_limit and error > desired_error:
    # this array will hold all errors from the current epoch
    errors = []
    # loop through all samples
    for sample_index in range(samples.shape[0]):
        # this is a bit tricky - samples[sample_index] returns vector, but we need a matrix, so we wrap it in array
        sample = numpy.array([samples[sample_index]])
        result = numpy.array([results[sample_index]])

        # Feed forward through both layers
        first_output = sigmoid(numpy.dot(sample, first_layer))
        second_output = sigmoid(numpy.dot(first_output, second_layer))

        # Compute output error and add the error to current epoch errors
        second_error = result - second_output

        errors.append(numpy.max(numpy.abs(second_error)))

        # the delta represents how much each of the weights contribute to the error
        second_delta = second_error * dsigmoid(second_output)

        # how much did each first layer value contribute to the second layer error (according to the weights)?
        first_error = second_delta.dot(second_layer.T)

        # the delta represents how much each of the weights contribute to the error
        first_delta = first_error * dsigmoid(first_output)

        second_layer += first_output.T.dot(second_delta) * rate
        first_layer += sample.T.dot(first_delta) * rate

    # select max error found during the epoch
    error = max(errors)

    # print current epoch status
    print('Epoch: %4d (of maximum %4d), max error: %.5f (of desired < %.5f)' % (epoch, epoch_limit, error, desired_error))
    epoch += 1

print('Actual testing of trained NN')

table_data = [
    ['Sample', 'Digit', 'Sample', 'Digit', 'Sample', 'Digit', 'Sample', 'Digit']
]

# we print three rows
for row in xrange(3):
    table_data.append([''] * 8)
    # with 8 columns, 4 image -> result pairs
    for col in xrange(4):
        # pick one random sample between 0 and sample count
        ri = random.randint(0, samples.shape[0] - 1)
        sample = samples[ri]

        table_data[row+1][col*2] = print_sample(sample)
        table_data[row+1][col*2+1] = '\n'.join([' ' * 5, ' ' * 5, '  %d' % compute_result(sample)])

table = AsciiTable(table_data)
table.inner_row_border = True

print(table.table)
