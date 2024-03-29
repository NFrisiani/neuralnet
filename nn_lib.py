import numpy as np
import pickle

# PREFILLED
def xavier_init(size, gain=1.0):
    """
    Xavier initialization of network weights.
    """
    low = -gain * np.sqrt(6.0 / np.sum(size))
    high = gain * np.sqrt(6.0 / np.sum(size))
    return np.random.uniform(low=low, high=high, size=size)

# PREFILLED
class Layer:
    """
    Abstract layer class.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, *args, **kwargs):
        raise NotImplementedError()

    def update_params(self, *args, **kwargs):
        pass

# PREFILLED
class MSELossLayer(Layer):
    """
    MSELossLayer: Computes mean-squared error between y_pred and y_target.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def _mse(y_pred, y_target):
        return np.mean((y_pred - y_target) ** 2)

    @staticmethod
    def _mse_grad(y_pred, y_target):
        return 2 * (y_pred - y_target) / len(y_pred)

    def forward(self, y_pred, y_target):
        self._cache_current = y_pred, y_target
        return self._mse(y_pred, y_target)

    def backward(self):
        return self._mse_grad(*self._cache_current)

# PREFILLED
class CrossEntropyLossLayer(Layer):
    """
    CrossEntropyLossLayer: Computes the softmax followed by the negative log-
    likelihood loss.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def softmax(x):
        numer = np.exp(x - x.max(axis=1, keepdims=True))
        denom = numer.sum(axis=1, keepdims=True)
        return numer / denom

    def forward(self, inputs, y_target):
        assert len(inputs) == len(y_target)
        n_obs = len(y_target)
        probs = self.softmax(inputs)
        self._cache_current = y_target, probs

        out = -1 / n_obs * np.sum(y_target * np.log(probs))
        return out

    def backward(self):
        y_target, probs = self._cache_current
        n_obs = len(y_target)
        return -1 / n_obs * (y_target - probs)

# DONE
class SigmoidLayer(Layer):
    """
    SigmoidLayer: Applies sigmoid function elementwise.
    """

    def __init__(self):
        self._cache_current = None

    def forward(self, x):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        return self.sigmoid(x)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        return self.sigmoid(grad_z) * (1 - self.sigmoid(grad_z))
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    @staticmethod
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

# DONE
class ReluLayer(Layer):
    """
    ReluLayer: Applies Relu function elementwise.
    """

    def __init__(self):
        self._cache_current = None

    def forward(self, x):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        return np.maximum(0.0, x)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z):
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        grad_z[grad_z >= 0] = 1.0
        grad_z[grad_z <  0] = 0.0
        return grad_z
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

# DONE
class LinearLayer(Layer):
    """
    LinearLayer: Performs affine transformation of input.
    """

    def __init__(self, n_in, n_out):
        """Constructor.

        Arguments:
            n_in {int} -- Number (or dimension) of inputs.
            n_out {int} -- Number (or dimension) of outputs.
        """
        self.n_in = n_in
        self.n_out = n_out

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        size = self.n_in * self.n_out
        self._W = xavier_init(size).reshape(self.n_in, self.n_out)
        self._b = xavier_init(self.n_out)

        self._cache_current  = {'x': None}
        self._grad_W_current = None
        self._grad_b_current = None

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def forward(self, x):
        """
        Performs forward pass through the layer (i.e. returns Wx + b).

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._cache_current['x'] = x
        # print(x)
        # print(x.shape, self._W.shape, self._b.shape)
        return np.dot(x, self._W) + self._b
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, n_in).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # get batch size
        batch = grad_z.shape[0]

        # Load forward propagation results
        x = self._cache_current['x']
        
        # calculate loss derivative with respect to weights
        dW = np.dot(x.T, grad_z) - self._W
        dW /= batch
        self._grad_W_current = dW

        # Calculate loss derivative with respect to bias
        db = np.mean(grad_z)
        self._grad_b_current = db

        # Calculate loss derivative with respect to input
        dx = np.dot(grad_z, dW.T)
        return dx
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        layer's parameters using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._b -= learning_rate * self._grad_b_current        
        self._W -= learning_rate * \
            np.array([np.mean(self._grad_W_current, axis=1)]).T
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

# DONE (Needs testing)
class MultiLayerNetwork(object):
    """
    MultiLayerNetwork: A network consisting of stacked linear layers and
    activation functions.
    """

    def __init__(self, input_dim, neurons, activations):
        """Constructor.

        Arguments:
            input_dim {int} -- Dimension of input (excluding batch dimension).
            neurons {list} -- Number of neurons in each layer represented as a
                list (the length of the list determines the number of layers).
            activations {list} -- List of the activation function to use for
                each layer.
        """
        self.input_dim = input_dim
        self.neurons = neurons
        self.activations = activations
        self._layers = []
        self._activations = []

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # add the input layer
        self._layers.append(LinearLayer(self.input_dim, self.neurons[0]))
        # setup the layers of the network.
        for i in range(1, len(self.neurons)):
            previous, current = self.neurons[i-1], self.neurons[i]
            self._layers.append(LinearLayer(previous, current))
        # add the activations
        for i in self.activations:
            if i == "relu":
                self._activations.append(ReluLayer())
            if i == "sigmoid":
                self._activations.append(SigmoidLayer())
            if i == "identity":
                self._activations.append(None)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def forward(self, x):
        """
        Performs forward pass through the network.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, input_dim).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size,
                #_neurons_in_final_layer)
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        for i in range(len(self._layers)):
            layer = self._layers[i]
            activation = self._activations[i]
            x = layer.forward(x)
            if activation is not None:
                x = activation(x)
        return x
        
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_z):
        """
        Performs backward pass through the network.

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (1,
                #_neurons_in_final_layer).

        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, input_dim).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        gradient = grad_z
        for i in range(len(self._layers)-1,-1,-1):
            activation = self._activations[i]
            if activation is not None:
                gradient = activation.backward(gradient)
            gradient = self._layers[i].backward(gradient)
        return gradient
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        parameters of all layers using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        for layer in self._layers:
            layer.update_params(learning_rate)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

# PREFILLED
def save_network(network, fpath):
    """
    Utility function to pickle `network` at file path `fpath`.
    """
    with open(fpath, "wb") as f:
        pickle.dump(network, f)

# PREFILLED
def load_network(fpath):
    """
    Utility function to load network found at file path `fpath`.
    """
    with open(fpath, "rb") as f:
        network = pickle.load(f)
    return network

# TODO: complete shuffle(), train(), eval_loss()
class Trainer(object):
    """
    Trainer: Object that manages the training of a neural network.
    """

    def __init__(
        self,
        network,
        batch_size,
        nb_epoch,
        learning_rate,
        loss_fun,
        shuffle_flag,
    ):
        """Constructor.

        Arguments:
            network {MultiLayerNetwork} -- MultiLayerNetwork to be trained.
            batch_size {int} -- Training batch size.
            nb_epoch {int} -- Number of training epochs.
            learning_rate {float} -- SGD learning rate to be used in training.
            loss_fun {str} -- Loss function to be used. Possible values: mse,
                bce.
            shuffle_flag {bool} -- If True, training data is shuffled before
                training.
        """
        self.network = network
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun
        self.shuffle_flag = shuffle_flag

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self._loss_layer = None
        self.lossFunction = None
        if self.loss_fun == "cross_entropy":
            self.lossFunction = CrossEntropyLossLayer()
        elif self.loss_fun == "MSE":
            self.lossFunction = MSELossLayer()
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    @staticmethod
    def shuffle(input_dataset, target_dataset):
        """
        Returns shuffled versions of the inputs.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_data_points, ).

        Returns: 2-tuple of np.ndarray: (shuffled inputs, shuffled_targets).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        index = np.argsort(input_dataset)
        index = np.random.shuffle(index)

        shuffled_inputs  =  input_dataset[index][0]
        shuffled_targets = target_dataset[index][0]

        return (shuffled_inputs, shuffled_targets)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def train(self, input_dataset, target_dataset):
        """
        Main training loop. Performs the following steps `nb_epoch` times:
            - Shuffles the input data (if `shuffle` is True)
            - Splits the dataset into batches of size `batch_size`.
            - For each batch:
                - Performs forward pass through the network given the current
                batch of inputs.
                - Computes loss.
                - Performs backward pass to compute gradients of loss with
                respect to parameters of network.
                - Performs one step of gradient descent on the network
                parameters.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_training_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_training_data_points, ).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        for i in range(self.nb_epoch):
            # setup loss function
            loss = 0
            # shuffle the dataset if needed.
            inputs, targets = input_dataset.copy(), target_dataset.copy()

            if self.shuffle_flag:
                inputs, targets = self.shuffle(inputs, targets)
            # split the dataset into batches.
            inputShape = inputs.shape[0]
            # print(inputs.shape)
            for j in range(0, inputShape, self.batch_size):
                X =  inputs[j:min(j+self.batch_size, inputShape), :]
                y = targets[j:min(j+self.batch_size, inputShape), :]
                # print("X:",X.shape, "y:",y.shape)
                # print("training network - line 491")
                # train the network
                guess = self.network(X)
                # check the loss
                # print("boss:",guess.shape,y.shape, loss)
                loss += self.lossFunction(y, guess)
                # get the loss derivative
                dLoss = self.lossFunction.backward()
                # backpropagate
                self.network.backward(dLoss)
                # update the weights.
                self.network.update_params(self.learning_rate)
            print("Epoch [{}/{}]: Loss:{}".format(i+1,self.nb_epoch,loss))
        print()
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def eval_loss(self, input_dataset, target_dataset):
        """
        Function that evaluate the loss function for given data.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_evaluation_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_evaluation_data_points, ).
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        guess = self.network(input_dataset)
        return self.lossFunction(target_dataset, guess)
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

# TODO: complete init(), apply(), revert()
class Preprocessor(object):
    """
    Preprocessor: Object used to apply "preprocessing" operation to datasets.
    The object can also be used to revert the changes.
    """

    def __init__(self, data):
        """
        Initializes the Preprocessor according to the provided dataset.
        (Does not modify the dataset.)

        Arguments:
            - data {np.ndarray} dataset used to determined the parameters for
            the normalization.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        self.data = data
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def apply(self, data):
        """
        Apply the pre-processing operations to the provided dataset.

        Arguments:
            - data {np.ndarray} dataset to be normalized.

        Returns:
            {np.ndarray} normalized dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        return data
        # norm=np.linalg.norm(data, ord=1)
        # if norm==0:
        #     norm=np.finfo(data.dtype).eps
        # return data/norm
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def revert(self, data):
        """
        Revert the pre-processing operations to retreive the original dataset.

        Arguments:
            - data {np.ndarray} dataset for which to revert normalization.

        Returns:
            {np.ndarray} reverted dataset.
        """
        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        return self.data
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

# PREFILLED
def example_main():
    input_dim = 4
    neurons = [16, 3]
    activations = ["relu", "sigmoid"]
    net = MultiLayerNetwork(input_dim, neurons, activations)

    dat = np.loadtxt("iris.dat")
    np.random.shuffle(dat)

    x = dat[:, :4]
    y = dat[:, 4:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    prep_input = Preprocessor(x_train)

    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    trainer = Trainer(
        network=net,
        batch_size=8,
        nb_epoch=1000,
        learning_rate=0.01,
        loss_fun="cross_entropy",
        shuffle_flag=True,
    )

    trainer.train(x_train_pre, y_train)
    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

    preds = net(x_val_pre).argmax(axis=1).squeeze()
    targets = y_val.argmax(axis=1).squeeze()
    accuracy = (preds == targets).mean()
    print("Validation accuracy: {}".format(accuracy))

# ---------------------

def testMe():
    lr = 0.001
    lossFunction = MSELossLayer()
    model = LinearLayer(7,1)
    relu = ReluLayer()
    x = np.array([[1,2,3,4,5,6,7], [1,2,3,4,5,6,7], [1,2,3,4,5,6,7]])
    y = np.array([1,1,1])
    for _ in range(0,100):
        # print("X:\n", x)
        # print("y:\n", y)
        guess = model.forward(x)
        # print("ok")
        guess = relu(guess)
        # print("Guess:\n",guess)
        loss = lossFunction(guess, y)
        # print(loss)
        # print("Loss:\n", loss)
        dLoss = lossFunction.backward()
        # print("DLoss:\n", dLoss)
        dloss = relu.backward(dLoss)

        model.backward(dLoss)
        model.update_params(lr)
    # print()
if __name__ == "__main__":
    example_main()
    # testMe()
