import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

'''
    Step 1. Data Preparation Stage 
'''
# generation some house sizes between 1000 and 3500 (typical sq ft of house)
num_house = 160

# np.random.seed(42)
# house_size = np.random.randint(low=1000, high=3500, size=num_house)
# print(house_size)

# generate house prices from house size with a random noise added
# np.random.seed(42)
# house_price = house_size * 100 + np.random.randint(low=20000, high=70000, size=num_house)
# print(house_price)

housing_data = pd.read_csv("..\\data\\train.csv", index_col="Id")
# print(housing_data.head)
house_size = housing_data['LotArea']
house_price = housing_data['SalePrice']

# plot the size vs price graph
plt.plot(house_size, house_price, "bx")  # bx = blue x
plt.ylabel('Price')
plt.xlabel('Size')
plt.show()


# normalize values to prevent under/over flow
def normalize(array):
    return (array - array.mean()) / array.std()


# Define number of training samples, 0.7 = 70%. We can take the first 70% since the values are randomized
# num_train_samples = math.floor(num_house * 0.7)
num_train_samples = housing_data.size

# define training data
train_house_size = np.asarray(house_size)
train_house_price = np.asarray(house_price)

print(train_house_size)

train_house_size_normalized = normalize(train_house_size)
train_house_price_normalized = normalize(train_house_price)

# define testing data
test_house_size = np.asarray(house_size[num_train_samples:])
test_house_price = np.asarray(house_price[num_train_samples:])

test_house_size_normalized = normalize(test_house_size)
test_house_price_normalized = normalize(test_house_price)

# Setup TensorFlow placeholders that get updated as we descent down the gradient
tf_house_size = tf.placeholder("float", name="house_size")
tf_house_price = tf.placeholder("float", name="price")

# Define the variables holding the size_factor and price we set during training
# We initialize them to some random values based on the normal distribution
tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
tf_premium = tf.Variable(np.random.randn(), name="premium")

'''
Step 2: Define inference function
'''
# We are assuming inference as below:
# price = (size * size_factor) + premium

# Define the operations for the predicting values
tf_price_pred = tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_premium)

'''
Step 3. Loss calculation
'''
# Define loss function -- mean squared error
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred - tf_house_price, 2)) / (2 * num_train_samples)

'''
Step 4: Optimize the algorithm
'''
# Optimize learning rate. The size of steps down the gradient
learning_rate = 0.1

# Define a Gradient Decent optimizer that will minimize the loss defined in the operation cost
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

'''
Step 5: Train Model
'''

# Initialize the variables
init = tf.global_variables_initializer()

# Launch the graph in the session
with tf.Session() as sess:
    sess.run(init)

    # set how often to display training progress and number of training iterations
    display_every = 2
    num_training_iter = 50

    # Calculate the number of lines to animation
    fit_num_plots = math.floor(num_training_iter / display_every)
    # Add storage of factor and offset values from each epoch
    fit_size_factor = np.zeros(fit_num_plots)
    fit_price_offsets = np.zeros(fit_num_plots)
    fit_plot_idx = 0

    # keep iterating the training data
    for iteration in range(num_train_samples):
        # Fit all training data
        for (x, y) in zip(train_house_size_normalized, train_house_price_normalized):
            sess.run(optimizer, feed_dict={tf_house_size: x, tf_house_price: y})

        # Display current status
        if (iteration + 1) % display_every == 0:
            c = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_normalized,
                                             tf_house_price: train_house_price_normalized})
            print("iteration #:", '%04d' % (iteration + 1), 'cost=', '{:.9f}'.format(c), 'size_factor=',
                  sess.run(tf_size_factor), "premium=", sess.run(tf_premium))

    print("Optimization Finished")
    training_cost = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_normalized,
                                                 tf_house_price: train_house_price_normalized})
    print("Trained Cost=", training_cost, "size_factor=", sess.run(tf_size_factor), "premium=", sess.run(tf_premium))

    # Plot of the training and test data, and learned regression

    # Get values sued to normalized data so we can denormalize data back to its original scale
    train_house_size_mean = train_house_size.mean()
    train_house_size_std = train_house_size.std()

    train_price_mean = train_house_price.mean()
    train_price_std = train_house_price.std()

    # Plot the graph
    plt.rcParams["figure.figsize"] = (10, 8)
    plt.figure()
    plt.ylabel('Price')
    plt.xlabel('Size (sq.ft)')
    plt.plot(train_house_size, train_house_price, 'go', label='Training data')
    plt.plot(test_house_size, test_house_price, 'mo', label='Testing data')
    plt.plot(train_house_size_normalized * train_house_size_std + train_house_size_mean,
             (sess.run(tf_size_factor) * train_house_size_normalized + sess.run(
                 tf_premium)) * train_price_std + train_price_mean,
             label='Learned Regression')

    plt.legend(loc='upper left')
    plt.show()

    # Plot another graph that animation of how Gradient Descent sequentially adjusted size_factor and price_offset to
    # find the values that returned the "best" fit line
    fig, ax = plt.subplots()
    line, = ax.plot(house_size, house_price)

    plt.rcParams['figure.figsize'] = (10, 8)
    plt.title('Gradient Descent Fitting Regression Line')
    plt.ylabel('Price')
    plt.xlabel('Size (sq.ft)')
    plt.plot(train_house_size, train_house_price, 'go', label='Training data')
    plt.plot(test_house_size, test_house_price, 'mo', label='Testing data')


    def animate(i):
        line.set_xdata(train_house_size_normalized * train_house_size_std + train_house_size_mean)
        line.set_ydata(
            (fit_size_factor[i] * train_house_size_normalized + fit_price_offsets[
                i]) * train_price_std + train_price_mean)
        return line,


    # Init only required for blitting to give a clean slate
    def initAnim():
        line.set_ydata(np.zeros(shape=house_price.shape[0]))  # set y's to 0
        return line,


    ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, fit_plot_idx), init_func=initAnim, interval=1000,
                                  blit=True)

    plt.show()
