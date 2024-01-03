# Basic Terms & Concepts (Good for revision)

**Gradients**: are calculated in nn to optimize the model's parameters. 

The goal of training a neural network is to minimize the difference between the "predicted output" and the "actual output".

**Loss function**: this difference is measured as a loss function.

**Back-Propagation**: The gradients of the loss function w.r.t model's parameters are used to update the parameters in the direction that minimizes the loss function. This process is called b-p.

**B-P working**: It works by propagating the error from the output layer back to input layer of nn.

=====================================
During this process, the gradients of loss function w.r.t are calculated using the chain rule of calculus. These gradients are then used to update the model's parameters using an optimization algorithm such as stochastic gradient descent.

Code:

```
import tensorflow as tf

# Define a neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Calculate gradients using backpropagation
with tf.GradientTape() as tape:
    logits = model(x_test)
    loss_value = tf.keras.losses.categorical_crossentropy(y_test, logits)

grads = tape.gradient(loss_value, model.trainable_weights)
```

In this example, we define a neural network using the tf.keras.Sequential API. We then compile the model using the compile method and train it using the fit method. Finally, we calculate the gradients of the loss function with respect to the model’s parameters using the tf.GradientTape context manager.

Adam is a predefined optimizer in TensorFlow. It is an extension of the stochastic gradient descent (SGD) algorithm that is widely used in deep learning. The Adam optimizer uses adaptive learning rates for each weight in the network and is computationally efficient. It is also well-suited for problems that are large in terms of data and/or parameters, and appropriate for non-stationary objectives and problems with very noisy or sparse gradients1.

In the code snippet I provided, we use the Adam optimizer to update the model’s parameters during training. The Adam optimizer is instantiated using the optimizer='adam' argument in the model.compile() method. This tells TensorFlow to use the Adam optimizer to minimize the loss function during training.
```
In the code snippet you provided, tf.keras.layers.Dense(64, activation='relu') creates a dense layer with 64 neurons and a rectified linear unit (ReLU) activation function. tf.keras.layers.Dense(10, activation='softmax') creates another dense layer with 10 neurons and a softmax activation function. These layers can be used to build a neural network for classification tasks.
```



**Q:How are parameters updated during backpropagation?**
A:  the parameters of a neural network are updated using the gradients of the loss function with respect to the parameters. The update rule for the parameters is typically of the form:

```
parameter = parameter - learning_rate * gradient
```
* where parameter is a weight or bias in the network, learning_rate is a hyperparameter that controls the step size of the update, and gradient is the gradient of the loss function with respect to the parameter.

Here is a table that shows the update rule for a single weight in a neural network:
-------------------------------------------------------------------------------------------------------------------------

| Parameter | Update Rule |
| :--: | :-: |
| Weight | weight = weight - learning_rate * gradient |


And here is a flowchart that shows the process of updating the parameters during backpropagation:
```
input data -> forward pass -> loss function -> backward pass -> gradients -> update parameters -> repeat
```

**Bias**: A bias is a scalar value that is added to the weighted sum of the inputs and weights of a neuron. It allows the neuron to shift the activation function to the left or right, which can be useful for modeling certain types of data.

**Learning rate**: The learning rate is a hyperparameter that controls the step size of the optimization algorithm during training. A higher learning rate can lead to faster convergence, but it can also cause the optimization algorithm to overshoot the minimum of the loss function.

**Weight**: A weight is a scalar value that is multiplied by the input of a neuron. It determines the strength of the connection between the input and the neuron. During training, the weights are updated using the gradients of the loss function with respect to the weights.

**Epochs**: An epoch is a complete pass through the training dataset during training. During each epoch, the model is trained on all of the training examples in the dataset.

**Batch size**: The batch size is the number of training examples used in each iteration of the optimization algorithm. During each iteration, the gradients of the loss function with respect to the weights are calculated using the batch of training examples.

**Activation**: An activation function is a function that is applied to the output of a neuron. It introduces non-linearity into the model, which can be useful for modeling complex data.
