This is a Artificial Neuronal Network made to predict if a person has or not Parkinson. It uses Linear Algebra in order to learn and find the most probable solution.

The ANN learns from a set of 5875 input data knowing that at the end of each row is the expected output for a person. This is done using Stathistical Normalization.

At the start we create random weights and activate each neuron with it's respective weight values. Activation is done using dot product between the input matrix and the weights matrix and then a sigmoid function is applied. This gives us the current output with the current weights. After we see the difference between what it is to be expected and what we actually get and apply a gradient descent adjustment and backpropage through the network.

It stops when the output data is at least 80% correct with the expected output data. This takes a little over 100 iterations.

This ANN is not very optimal producing a lot of useless noise when predicting if a person has or not Parkinson, but it does predict correctly. The noise is easy to be recognised since it's just a float of ether 1 or 0, forced there from the dot product requirement of the nr of lines in the first matrix being the same as the nr of columns in the second matrix.
