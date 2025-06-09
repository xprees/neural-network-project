## Neural Network Project (PV021) -- Deep Learning from Scratch

Project implements a neural network in low-level manner (without use of advanced libraries or frameworks) using **.NET 8** supporting native AOT
compilation, but JIT compilation seems to be more performant out-of-the box.

Project implements **MLP architecture** of Neural Network in its core to solve **FashionMNIST problem** with accuracy of **88%**.

### HOW TO RUN

First make the script executable, by setting proper permission

```shell
chmod u+x run.sh
```

Finally, run the whole neural-network **training + evaluation**
Beware that this will take a while, depending on your machine performance. (around 15 mins on i7-13700H CPU + 32GB RAM)

```shell
./run.sh
```

#### OUTPUT

- Your solution must output two files to the root project directory:
  (next to `example_test_predictions.csv` file):
    - `train_predictions.csv` - your network predictions for the train set.
    - `test_predictions.csv`  - your network predictions for the test set.
- The format of these files has to be the same as the supplied
  training/testing labels:
    - One prediction per line.
    - Prediction for i-th input vector (ordered by the input .csv file)
      must be on i-th line in the associated output file.
    - Each prediction is a single integer 0 - 9.

### DATASET

Fashion MNIST (https://arxiv.org/pdf/1708.07747.pdf) - a modern version of a
well-known MNIST (http://yann.lecun.com/exdb/mnist/). It is a dataset of
Zalando's article images ‒ consisting of a training set of 60,000 examples
and a test set of 10,000 examples. Each example is a 28x28 grayscale image,
associated with a label from 10 classes. The dataset is in CSV format. There
are four data files included:

- `fashion_mnist_train_vectors.csv`   - training input vectors
- `fashion_mnist_test_vectors.csv`    - testing input vectors
- `fashion_mnist_train_labels.csv`    - training labels
- `fashion_mnist_test_labels.csv`     - testing labels
