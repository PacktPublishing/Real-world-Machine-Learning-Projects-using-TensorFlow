
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import random 
import cv2
from tensorflow.contrib.layers import flatten
import tensorflow as tf
import csv
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import time
import matplotlib.gridspec as gridspec
import cv2
import pickle


# In[ ]:


# TODO: Fill this in based on where you saved the training and testing data
_
training_file = "./train.p"
testing_file = "./test.p"


# In[ ]:




with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train_o, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']



# In[ ]:


### Replace each question mark with the appropriate value.

# TODO: Number of training examples
n_train = len(X_train_o)

# TODO: Number of testing examples.
n_test = len(X_test)
# TODO: What's the shape of an traffic sign image?
image_shape = (X_train_o[0].shape)

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
#count of each type image
unique, index, counts = np.unique(y_train, return_index = True, return_counts=True)
#print (np.asarray((unique, counts, index)).T.shape)
#print (np.sum(counts))
#print (np.asarray((unique, index, counts)).T)


# In[ ]:


#count of each type image
unique, index, counts = np.unique(y_train, return_index = True,
                                  return_counts=True)
print ("unique", unique)
print ("counts", counts)
print ("index", index)

# print (np.asarray((unique, counts, index)).T.shape)
print (np.sum(counts))
print (np.asarray((unique, index, counts)).T)


# In[ ]:


#Plotting the bar graph of the frequency of classes 
plt.bar(unique, counts, 1)
plt.title('Class Frequency')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()


# In[ ]:


### Data exploration visualization goes here.
### Feel free to use as many code cells as needed.

# Visualizations will be shown in the notebook.

index1 = random.randint(0,len(X_train_o))
image1 = X_train_o[index1].squeeze()
index2 = 19
image2 = X_train_o[index2].squeeze()


plt.figure(figsize = (1,1))
plt.imshow(image1, cmap = 'gray')
print (y_train[index1])
plt.figure(figsize = (1,1))
plt.imshow(image2, cmap = 'gray')
print (y_train[index2])


# In[ ]:


#See the different class of signs 

with open('signnames.csv', 'r') as f:
    reader = csv.reader(f)
    class_names = dict(reader)

# choose one sample to use for visualization from each class
sample_class_image = []
for n in range(n_classes):
    sample_class_image.append(np.random.choice(np.where(y_train==n)[0]))


show_samples = X_train_o[sample_class_image,:,:,:]


# In[ ]:


## plot classes in a grid

# function to plot sample images in a grid
def plot_classes(box, grid_w, grid_h, stitch_layers=True):
    fig = plt.figure()
    for j in range(box.shape[0]):
        ax = fig.add_subplot(grid_h, grid_w, j+1)
        ax.imshow(box[j].squeeze(), cmap = 'gray')
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.tight_layout()


# In[ ]:


plot_classes(show_samples, 8,6)


# In[ ]:


# convert the the training images to grayscale
X_train = []
for i in range (np.shape(X_train_o)[0]):
    img = cv2.cvtColor(X_train_o[i,:,:,:], cv2.COLOR_BGR2GRAY)
    img = img.squeeze()
    img = img.reshape(32,32,1)
    X_train.append(img)
    
# show a sample image converted
plt.imshow(X_train[1000].squeeze(), cmap = 'gray')
plt.title('Sample Gray Image')
plt.show()


# In[ ]:


#Transform the image
def transform_image(img,ang_range,shear_range,trans_range):
    # Rotation
    # generating ang_rotation rondamly and let this angle takes - values 
    # by subtracting it from the ang_range it self 
    ang_rot = np.random.uniform(ang_range)-ang_range/2
    #print (img.shape)
    # then we will sort the rows, cols, ch from the shape of the images 
    # becuase those parameters are feed to the rotation Translation Shear
    #funtion using the
    # using the opencv library 
    rows,cols,ch = img.shape
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    # same as here we will generate a rondom values for the transilation 
    # as we know our image is gray scale image that means a 2D matrix 
    # that means the transilation will be only  x band y direction 
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)
        
    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))
    
    return img


# In[ ]:


import random
total_images_per_set = 2500
for i in range(n_classes):
    print (i),
    to_loop = int((total_images_per_set-counts[i])/100)
#    print (to_loop+1)
#    print (index[i])
#    print (counts[i])
    if to_loop < 0:
        continue
    for j in range(to_loop+1):
        m = random.randint(index[i],index[i] + counts[i]-1)
#        print ("m = ",m)
        index2 = m
        #print (X_train[index2].shape)
        #image2 = X_train[index2].squeeze()
        image2 = X_train[index2]
        for k in range(100):
            img = transform_image(image2,10,10,5)
            img = img.reshape(1,32,32,1)
            X_train = np.concatenate((X_train,img), axis = 0)
            y_train = np.append(y_train,unique[i])

print ("Transformation process completed")


# In[ ]:


#check the size of the data and counts of X_train_augmented
n_train = len(X_train)

# TODO: Number of testing examples.
n_test = len(X_test)
print (X_train.shape)
print (y_train.shape)
# TODO: What's the shape of an traffic sign image?
image_shape = (X_train[0].shape)

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
#count of each type image
unique, index, counts = np.unique(y_train, return_index = True, return_counts=True)


# In[ ]:


plt.bar(unique, counts, 1)
plt.title('Class Frequency')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()


# In[ ]:


#save the data in the file for future use
train_augmented_file = "./traffic-signs-data/train_augmented_gray.p" 
output = open(train_augmented_file, 'wb')

mydict2 = {'features': 1, 'labels': 2}
mydict2['features'] = X_train
mydict2['labels'] = y_train
pickle.dump(mydict2, output)
output.close()


# In[ ]:


# normalize dataset
X_train = (np.array(X_train) - 128.0)/256.0


# In[ ]:


### Generate data additional data (OPTIONAL!)
### and split the data into training/validation/testing sets here.
### Feel free to use as many code cells as needed.

X_train, y_train = shuffle(X_train,y_train)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)

print (len(X_train), len(y_train))
print (len(X_validation),len(y_validation))


# In[ ]:


EPOCHS = 5
BATCH_SIZE = 256

def SignTraffic(x, keep_prob):
    mu = 0
    sigma = 0.1
    #Layer1 32*32*1 --> 28*28*16 
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5,5,1,16), mean = mu, stddev = sigma), name = 'conv1_W')
    conv1_b = tf.Variable(tf.constant(0.1, shape = [16]), name = 'conv1_b')
    conv1 = tf.add(tf.nn.conv2d(x,conv1_W, strides = [1,1,1,1], padding = 'VALID') , conv1_b)
    #Layer1 activation 
    conv1 = tf.nn.relu(conv1)
    #Layer1 pooling 28*28*16 --> 14*14*16
    conv1 = tf.nn.max_pool(conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    
    #Layer2  14*14*16 --> 10*10*32
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5,5,16,32), mean = mu , stddev = sigma), name = 'conv2_W')
    conv2_b = tf.Variable(tf.constant(0.1, shape = [32]), name = 'conv2_b')
    conv2 = tf.add(tf.nn.conv2d(conv1, conv2_W, strides = [1,1,1,1], padding = 'VALID') , conv2_b)
    #Layer2 activation
    conv2 = tf.nn.relu(conv2)
    #Later2 pooling 10*10*32 --> 5*5*32 
    conv2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    
    #Flatten 5*5*32 --> 800 
    fc0 = flatten(conv2)
    
    #Layer3  800--> 516
    fc1_W = tf.Variable(tf.truncated_normal(shape = (800,516), mean = mu , stddev = sigma), name = 'fc1_W')
    fc1_b = tf.Variable(tf.constant(0.1, shape = [516]), name = 'fc1_b')
    fc1 = tf.add(tf.matmul(fc0, fc1_W) , fc1_b)
    #Layer3 activation
    fc1 = tf.nn.relu(fc1)
    
    #Dropout 
    fc1 = tf.nn.dropout(fc1,keep_prob)
    
    
    #Layer4 516--> 360
    fc2_W = tf.Variable(tf.truncated_normal(shape = (516,360), mean = mu , stddev = sigma), name = 'fc2_W')
    fc2_b = tf.Variable(tf.constant(0.1, shape = [360]), name = 'fc2_b')
    fc2 = tf.add(tf.matmul(fc1,fc2_W) , fc2_b)
    #Layer4 activation
    fc2 = tf.nn.relu(fc2)
    
    #Dropout 
    fc2 = tf.nn.dropout(fc2,keep_prob)
    
    #Layer5 360--> 43
    fc3_W = tf.Variable(tf.truncated_normal(shape = (360,43), mean = mu, stddev = sigma), name = 'fc3_W')
    fc3_b = tf.Variable(tf.constant(0.1, shape =[43]), name = 'fc3_b')
    logits = tf.add(tf.matmul(fc2,fc3_W) , fc3_b)
    
    return logits 
    
    


# In[ ]:


### Train your model here.
### Feel free to use as many code cells as needed.

save_file = 'saved/sign_classify.ckpt'

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y,43)


# In[ ]:


#Traning
starttime = time.clock()
rate = 0.001
logits = SignTraffic(x,keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=one_hot_y)
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits( logits=prediction, labels=target_output)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)


# In[ ]:


#Model Evaluation 
correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(one_hot_y,1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
saver = tf.train.Saver()
all_saver = tf.train.Saver(save_relative_paths=True)
# saver = tf.train.Saver(save_relative_paths=True)
def evaluate(X_data, y_data, keep_prob):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0,num_examples,BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict = {x:batch_x, y:batch_y, keep_prob : 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# In[ ]:


#Train the model 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    print ("Training...")
    
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train,y_train)
        for offset in range (0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset : end], y_train[offset:end]
            sess.run(training_operation, feed_dict = {x: batch_x, y: batch_y, keep_prob : 0.25})
        validation_accuracy = evaluate(X_validation, y_validation, keep_prob)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        


    print("Model saved")

    
endtime = time.clock()
print("execution took",endtime-starttime,"seconds")


# In[ ]:


print("Test Accuracy = {:.3f}".format(validation_accuracy))


# In[ ]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    path=saver.save(sess,"saved/sign_classify.ckpt" )

    print("Model saved")


# In[ ]:


#Test the model here, clear the previous model

drop_prob = 1.0
#convert to gray scale images
X_test_gray = []
for i in range (np.shape(X_test)[0]):
    img = cv2.cvtColor(X_test[i,:,:,:], cv2.COLOR_BGR2GRAY)
    img = img.squeeze()
    img = img.reshape(32,32,1)
    X_test_gray.append(img)

# normalize dataset [naive]
X_test_gray = (np.array(X_test_gray) - 128.0)/256.0

print (X_test.shape)
print (X_test_gray.shape)


# In[ ]:



with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())

#     saver.restore(sess, "C:/Users/El Holandy/AppData/Roaming/SPB_Data/saved_data_sign_class/data_lastt.ckpt"  )

    saver.restore(sess, "saved/sign_classify.ckpt"  )

    test_accuracy = evaluate(X_test_gray, y_test, keep_prob )
    print("Test Accuracy = {:.3f}".format(test_accuracy))
    
    


# In[ ]:


no_of_imgs = 25
disp_imgs = []
disp_imgs_gray = []
for n in range(no_of_imgs):
    image = cv2.imread('model_traffic_signs/'+str(n+1)+'.jpg')
    dim = (32,32)

    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    
    disp_imgs.append(np.asarray(resized))
    
    resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    disp_imgs_gray.append(np.asarray(resized))

# normalize new test data
test_imgs_gray = ((np.array(disp_imgs_gray)-128.0)/256.0).reshape(no_of_imgs,32,32,1)
#See the different class of signs 

with open('signnames.csv', 'r') as f:
    reader = csv.reader(f)
    class_names = dict(reader)

with tf.Session() as sess:
#     new_saver = tf.train.import_meta_graph('sign_classify.ckpt.meta')
#"C:/Users/El Holandy/AppData/Roaming/SPB_Data/saved_data_sign_class/data_lastt.ckpt" 
#     new_saver = tf.train.import_meta_graph('./saved_data_sign_class/data_lastt.ckpt.meta')
    new_saver = tf.train.import_meta_graph('./saved/sign_classify.ckpt.meta')

    new_saver.restore(sess, tf.train.latest_checkpoint('.'))
    
    # model evaluation
    prediction = tf.argmax(logits, 1)

    test_prediction = sess.run(
        prediction,
        feed_dict={x: test_imgs_gray, keep_prob: drop_prob})
for i in range(no_of_imgs):
    if i%5 == 0:
        print (" ")
    print('Prediction: {} \t| {}'.format(test_prediction[i], 
                                            class_names[str(test_prediction[i])]))
### Visualize the softmax probabilities here.
### Feel free to use as many code cells as needed.
# get the softmax probabilities for 3 best prediction probabilities.
with tf.Session() as sess:
#     new_saver = tf.train.import_meta_graph('./saved_data_sign_class/data_lastt.ckpt.meta')
    new_saver = tf.train.import_meta_graph('./saved/sign_classify.ckpt.meta')

    new_saver.restore(sess, tf.train.latest_checkpoint('.'))
    
    # model evaluation
    prediction = tf.nn.softmax(logits)

    test_prediction = sess.run(tf.nn.top_k(
        prediction,k=5),
        feed_dict={x: test_imgs_gray , keep_prob: drop_prob})

#print('Predictions: {}'.format(test_prediction))
# plot visualization of softmax probabilities
index = np.arange(5)
probabilities, predict_classes = test_prediction

candidates = [4,5,12,15,22]
for i,im in enumerate(candidates):
    
    plt.subplot(2,1,1)
    plt.imshow(disp_imgs[im])
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    
    plt.subplot(2,1,2)
    plt.barh(index, probabilities[im], height=0.5, align='center')
    plt.yticks(index,[class_names[str(predict_classes[im][j])] for j in index] )
    plt.show()


# In[ ]:


### Visualize the softmax probabilities here.
### Feel free to use as many code cells as needed.
# get the softmax probabilities for 3 best prediction probabilities.
with tf.Session() as sess:
#     new_saver = tf.train.import_meta_graph('./saved_data_sign_class/data_lastt.ckpt.meta')
    new_saver = tf.train.import_meta_graph('./saved/sign_classify.ckpt.meta')

    new_saver.restore(sess, tf.train.latest_checkpoint('.'))
    
    # model evaluation
    prediction = tf.nn.softmax(logits)

    test_prediction = sess.run(tf.nn.top_k(
        prediction,k=5),
        feed_dict={x: test_imgs_gray , keep_prob: drop_prob})

#print('Predictions: {}'.format(test_prediction))


# In[ ]:


# plot visualization of softmax probabilities
index = np.arange(5)
probabilities, predict_classes = test_prediction

candidates = [4,5,12,15,22]
for i,im in enumerate(candidates):
    
    plt.subplot(2,1,1)
    plt.imshow(disp_imgs[im])
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    
    plt.subplot(2,1,2)
    plt.barh(index, probabilities[im], height=0.5, align='center')
    plt.yticks(index,[class_names[str(predict_classes[im][j])] for j in index] )
    plt.show()

