import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_ENABLE_WINOGRAD_NONE_USED']='1'

# Hyper Parameters
learning_rate = 0.01
training_epochs = 1
batch_size = 100
tf.set_random_seed(25)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist", one_hot=True,reshape=False,validation_size=0)

# Step 1: Initial Setup
X = tf.placeholder(tf.float32, [None, 28, 28, 1], name='X')
y = tf.placeholder(tf.float32, [None, 10])
#pkeep = tf.placeholder(tf.float32,name='pkeep')

L1 = 4 # first convolutional filters
L2 = 8 # second convolutional filters
L3 = 12 # third convolutional filters
L4 = 200 # fully connected neurons

W1 = tf.Variable(tf.truncated_normal([5,5,1,L1], stddev=0.1))
B1 = tf.Variable(tf.truncated_normal([L1],stddev=0.1))
W2 = tf.Variable(tf.truncated_normal([3,3,L1,L2], stddev=0.1))
B2 = tf.Variable(tf.truncated_normal([L2],stddev=0.1))
W3 = tf.Variable(tf.truncated_normal([3,3,L2,L3], stddev=0.1))
B3 = tf.Variable(tf.truncated_normal([L3],stddev=0.1))
W4 = tf.Variable(tf.truncated_normal([7*7*L3,L4], stddev=0.1))
B4 = tf.Variable(tf.truncated_normal([L4],stddev=0.1))
W5 = tf.Variable(tf.truncated_normal([L4, 10], stddev=0.1))
B5 = tf.Variable(tf.truncated_normal([10],stddev=0.1))

# Step 2: Setup Model
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME') + B1)# output is 28x28
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1,1,1,1], padding='SAME') + B2)
#Y2= tf.nn.dropout(Y2, pkeep)
Y2 = tf.nn.max_pool(Y2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # output is 14x14
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1,1,1,1], padding='SAME') + B3)
#Y3= tf.nn.dropout(Y3, pkeep)
Y3 = tf.nn.max_pool(Y3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # output is 7x7

# Flatten the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 7 * 7 * L3])

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
#YY4 = tf.nn.dropout(Y4, pkeep)
Ylogits = tf.matmul(Y4, W5) + B5
yhat = tf.nn.softmax(Ylogits, name ='yhat')

# Step 3: Loss Functions
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=y))

# Step 4: Optimizer
#optimizer = tf.train.GradientDescentOptimizer(learning_rate)
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# accuracy of the trained model
is_correct = tf.equal(tf.argmax(y,1),tf.argmax(yhat,1))
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Step 5: Training Loop
for epoch in range(training_epochs):
    num_batches = int(mnist.train.num_examples / batch_size)
    for i in range(num_batches):
        batch_X, batch_y = mnist.train.next_batch(batch_size)
        train_data = {X: batch_X, y: batch_y}
        sess.run(train, feed_dict=train_data)
        print(epoch * num_batches + i + 1, "Training accuracy =", sess.run(accuracy, feed_dict=train_data),
          "Loss =", sess.run(loss, feed_dict=train_data))

# Step 6: Evaluation
test_data = {X:mnist.test.images,y:mnist.test.labels}
print("Testing Accuracy = ", sess.run(accuracy, feed_dict = test_data))

saver = tf.train.Saver()
saver.save(sess, "./models/tmp_cnn/mnist.ckpt")