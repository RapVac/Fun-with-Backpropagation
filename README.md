# Fun with Backpropagation

A series of basic projects to build backpropagating neural networks from scratch.

``backprop.pdf`` and ``src.text`` are the formatted and LaTeX explanations for matrix backpropagation.

### 3 In a Row

Simple neural net designed to identify 3 objects in a row on a 3x3 grid. Served as a prototype that was then streamlined for image recognition.

Included files:

| Name | Description |
| :--- | :---------- |
| 3InARow_neural_net.py | Neural network + backpropagation math. |
| 3InARow_training_data | All possible ways, labeled, to arrange objects on a 3x3 grid. |
| generate_3InARow.py | Short script to generate the training data. |

### Shape Recognition

Larger neural net, built on the same principles as the 3 In a Row network. Identifies 65x65 pixel images as containing one of either a cube, cone, square pyramid, sphere, cylinder, or tetrahedron. Achieved decent performance with two layers of 80 neurons, training on 12,000 images for 24 hours.

Included files:

| Name | Description |
| :--- | :---------- |
| image_recognition.py | Neural network + backpropagation math. |
| visualize.py | Short script to visualize the weights going from the input layer to the first hidden layer. |
| training_images -> training_data.7z | 7z archive containing the 12000 used training images. Images were randomly generated using the tkinter3d project. |
| testing_images -> * | A series of random images for testing the trained network. |
