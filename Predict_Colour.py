import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
Predict colour of square in 28, 28 image
'''
################CREATE DATASET#####################
colors=[10,20,30,40,50,60,70,80,90,100] 

def create_dataset(samples,colors=[10,20,30,40,50,60,70,80,90,100] ,split=0.7):
  dataset = []
  labels  = []
  for i in range(samples):
    dataset.append(np.random.randint(0,255,size=(28,28)))
    start = np.random.randint(5,10)
    end   = np.random.randint(14, 28)
    colour = np.random.choice(colors)
    labels.append(colors.index(colour))
    for row in range(start,end):
      for col in range(start,end):
        dataset[i][row,col] = colour
  X_train = np.array(dataset[:int(samples*split)])
  y_train = np.array(labels[:int(samples*split)])
  X_test = np.array(dataset[int(samples*split):])
  y_test = np.array(labels[int(samples*split):])
  return X_train/255.0, y_train, X_test/255.0, y_test

X_train, y_train, x_test, y_test = create_dataset(10000, colors)
plt.figure()
plt.title(f"{y_train[1]}")
plt.imshow(X_train[1])
plt.colorbar()
plt.grid(False)
plt.show()

################CREATE NETWORK#####################

# define metrics
def Recall(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall

def Precision(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision

def F1(y_true, y_pred):
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+tf.keras.backend.epsilon()))

n_features = len(colors)
### layer input
inputs = tf.keras.layers.Input(name="input", shape=(28,28))
flat_inputs = tf.keras.layers.Flatten(input_shape=(28,28))(inputs)
### hidden layer 1
h1 = tf.keras.layers.Dense(name="h1", units=64, activation='relu')(flat_inputs)
### Dropout layer 1
d1 = tf.keras.layers.Dropout(name="drop1", rate=0.2)(h1)
### hidden layer 2
h2 = tf.keras.layers.Dense(name="h2", units=64, activation='relu')(d1)
### Dropout layer 1
d2 = tf.keras.layers.Dropout(name="drop2", rate=0.2)(h2)
### hidden layer 2
h3 = tf.keras.layers.Dense(name="h3", units=10)(d2)
#h2 = tf.keras.layers.Dropout(name="drop2", rate=0.2)(h2)
### layer output
outputs = tf.keras.layers.Dense(name="output", units=10, activation='softmax')(h3)
model = tf.keras.Model(inputs=inputs, outputs=outputs, name="DeepNN")
model.compile(optimizer='SGD',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy',F1])

################TRAIN NETWORK#####################

# train/validation
training = model.fit(X_train, y_train, batch_size=32, epochs=100, shuffle=True)

# plot
metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]    
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15,3))
       
## training    
ax[0].set(title="Training")    
ax11 = ax[0].twinx()    
ax[0].plot(training.history['loss'], color='black'),ax[0].set_xlabel('Epochs')    
ax[0].set_ylabel('Loss', color='black')    
for metric in metrics:        
    ax11.plot(training.history[metric], label=metric)
    ax11.set_ylabel("Score", color='steelblue')    
ax11.legend()

plt.show()


################PLOT PREDICTIONS#####################


def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(colors[predicted_label],
                                100*np.max(predictions_array),
                                colors[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

i = 10
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], y_test, x_test)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  y_test)
plt.show()
