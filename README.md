# M-L-notes

Gradients: are calculated in nn to optimize the model's parameters. 

The goal of training a neural network is to minimize the difference between the "predicted output" and the "actual output".

Loss funciton: this difference is measured as a loss function.

Back-Propagation: The gradients of the loss function w.r.t model's parameters are used to update the parameters in the direction that minimizes the loss function. This process is called b-p.

B-P working: It works by propagating the error from the output layer back to input layer of nn.

During this process, the gradients of loss function w.r.t are calculated using the chain rule of calculus. These gradients are then used to update the model's parameters using an optimization algorithm such as stochastic gradient descent.

Code:
=====================================
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
=======================================

In this example, we define a neural network using the tf.keras.Sequential API. We then compile the model using the compile method and train it using the fit method. Finally, we calculate the gradients of the loss function with respect to the model’s parameters using the tf.GradientTape context manager.


-----------*****************-----------


Q:How are parameters updated during backpropagation?
A:  the parameters of a neural network are updated using the gradients of the loss function with respect to the parameters. The update rule for the parameters is typically of the form:

Code:
parameter = parameter - learning_rate * gradient

* where parameter is a weight or bias in the network, learning_rate is a hyperparameter that controls the step size of the update, and gradient is the gradient of the loss function with respect to the parameter.

Here is a table that shows the update rule for a single weight in a neural network:
-------------------------------------------------------------------------------------------------------------------------
| Parameter	                                             Update Rule                                           |
| Weight	                              weight = weight - learning_rate * gradient               |
-------------------------------------------------------------------------------------------------------------------------

And here is a flowchart that shows the process of updating the parameters during backpropagation:

input data -> forward pass -> loss function -> backward pass -> gradients -> update parameters -> repeat

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
