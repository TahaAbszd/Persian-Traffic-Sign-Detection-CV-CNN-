import tensorflow as tf
from tensorflow.keras import layers , models
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

image_size = 32
no_of_classes = 43
image_shape = (32,32,3)


model = models.Sequential([layers.Conv2D(32 , (3,3) , activation='relu' , padding='same' , input_shape = (32,32,3)),
                           layers.MaxPool2D((2,2)),
                           layers.Conv2D(64 , (3,3) , activation='relu' , padding='same'),
                           layers.MaxPool2D((2,2)),
                           layers.Conv2D(128, (3,3) , activation='relu'),
                           layers.Flatten(),
                           layers.Dense(128, activation='relu'),
                           layers.Dense(43 , activation='softmax')
                           ])

opt = Adam(learning_rate= 0.001)

model.compile(optimizer =opt , loss = 'categorical_crossentropy', metrics= ['accuracy'])
# model.summary()

#test
x_test = np.random.rand(10, 32,32,3)
y_test = np.zeros((10,43))
y_test[np.arange(10), np.random.randint(0,43,10)] = 1.0

history = model.fit(x_test,y_test , epochs = 2 , batch_size = 2)
print('tested successfuly')



# train = model.fit(x = train_batches , validation_data=(val_batches) ,  epochs = 20 , verbose = 2)

# model.save('CNNModel.h5',include_optimizer=True)