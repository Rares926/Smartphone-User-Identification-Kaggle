import tensorflow as tf
import numpy as np 
import pandas as pd 
import os 
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn import metrics


print("Number of available GPU's: ", len(tf.config.list_physical_devices('GPU')))

def generate_subbmision_file(model_output, data:str="dataset/test", id=1):
    """
    Generates a .txt file that contains the predicted labels on the test data.

    Args:
        model_output : The predictions.
        data (str, optional): Path to the test data.
        id (int, optional): Id to be used inside submission file name.
    """
    predicted_labels = [np.argmax(prediction) + 1 for prediction in model_output]
    file_names = [file_name[:len(file_name)-4] for file_name in os.listdir(data)]
    column_values = ['id', 'class']
    df = pd.DataFrame(data = list(zip(file_names,predicted_labels)), 
                      columns = column_values)
    df.to_csv(f"submission_Patrascu_Rares_{id}.csv",index=None)


def load_labels(data_path: str):
    """
    Loads the labels using pandas.

    Args:
        data_path (str): Path to the csv containing class values.

    """
    labels =  pd.read_csv(data_path)
    preproc_labels = labels.values[:,1].reshape(len(labels),1) - 1
    return np.array(preproc_labels)

def load_data(data_path: str, column_size: int= 150):
    """
    Loads the training data in a numpy object. This method also computes the mean of all column sizes and prints it.

    Args:
        data_path (str): Path to the training data.
        column_size (int, optional): All the column values of a grid will be adjusted to this value. Defaults to 150.

    """
    test_data = []
    files = [data_path + file_name for file_name in os.listdir(data_path)]
    all_lenghts = []

    def preproc_input(input):
        input_shape = input.shape
        input =input.T
        tmp_testing_re = []
        for i in range(3):
            if len(input[i]) < column_size:
                    nr_of_0_to_append = column_size - len(input[i])
                    to_append = np.zeros(nr_of_0_to_append)
                    tmp_testing_re.append(np.concatenate((input[i],to_append.astype(np.float32))))
            else:
                tmp_testing_re.append(input[i][0:column_size])

        return np.array(tmp_testing_re).T

    for file in files: 
        print(file)
        one_input = pd.read_csv(file, header=None, names=['x', 'y', 'z']).values
        print(one_input.shape)
        all_lenghts.append(len(one_input))
        one_input = preproc_input(one_input)
        test_data.append(np.array(one_input))

    print(np.mean(all_lenghts))
    return np.array(test_data)


train_labels = load_labels("dataset/train_labels.csv")
test_data = load_data("dataset/test/", 150)
train_data = load_data("dataset/train/", 150)

train_data_cpy = train_data[:,:,:,np.newaxis]
test_data_cpy = test_data[:,:,:,np.newaxis]


each_fold_accuraccy, each_fold_loss = [], []


kfold = KFold(n_splits=5, shuffle=True)
fold_index = 1
kf_number = 10

for train, test in kfold.split(train_data_cpy, train_labels):

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),input_shape = (150,3,1), activation='linear', strides=(1,1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='linear', strides=(1,1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='linear', strides=(1,1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
        tf.keras.layers.Conv2D(filters=248, kernel_size=(3, 3), activation='linear', strides=(1,1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
        tf.keras.layers.Conv2D(filters=248, kernel_size=(3, 3), activation='linear', strides=(1,1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
        tf.keras.layers.Conv2D(filters=496, kernel_size=(3, 3), activation='linear', strides=(1,1), padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.GlobalAveragePooling2D(data_format=None, keepdims=False),
        tf.keras.layers.Dense(50, activation='linear'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(20, activation='softmax')
        ])

    # Models we're usually tested with an exponential 

    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     0.001,
    #     decay_steps=100000,
    #     decay_rate=0.96,
    #     staircase=True)

    # optimizer = SGD(learning_rate=lr_schedule, momentum=0.9)  

    optimizer = SGD(learning_rate=0.001, momentum=0.9)  

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',  # used because we did not one-hot encode the labels
                  metrics=['accuracy'])


    print('#####################################################################')
    print('Fold {} started training....'.format(fold_index))

    model_checkpoint_callback = ModelCheckpoint(
        filepath= f'checkpoints/cv_{kf_number}/{fold_index}/best_checkpoint.ckpt',
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose = True)

    def scheduler_100_epochs(epoch, lr): # tested all models for 100 epochs 
        if epoch < 30:
            return lr
        elif epoch < 50:
            return lr * tf.math.exp(-0.1)
        elif epoch < 70:
            return lr * tf.math.exp(-0.01)
        elif epoch < 80:
            return lr * tf.math.exp(-0.001)
        else :
            return lr 

    def scheduler_100_epochs(epoch, lr): # tested all models for 100 epochs
        if epoch < 60:
            return lr
        elif epoch < 70:
            return lr * tf.math.exp(-0.1)
        elif epoch < 80:
            return lr * tf.math.exp(-0.01)
        elif epoch < 90:
            return lr * tf.math.exp(-0.001)
        else :
            return lr 

    def scheduler_50_epochs(epoch, lr): # used to test for bigger models for 50 epochs
        if epoch < 20:
            return lr
        elif epoch < 30:
            return lr * tf.math.exp(-0.1)
        elif epoch <= 50:
            return lr * tf.math.exp(-0.01)
        else :
            return lr

    lr_schedule_callback = tf.keras.callbacks.LearningRateScheduler(scheduler_100_epochs)


    history = model.fit(train_data_cpy[train], train_labels[train],
                        initial_epoch=0,
                        batch_size=32,
                        epochs=100,
                        validation_data=[train_data_cpy[test], train_labels[test]],
                        callbacks=[lr_schedule_callback, model_checkpoint_callback])


    model.load_weights(f"Z:/Master I/PML - Practical Machine Learning/Kaggle_Project/checkpoints/cv_{kf_number}/{fold_index}/best_checkpoint.ckpt")
    accurracy_and_loss = model.evaluate(train_data_cpy[test], train_labels[test], verbose=0)
    each_fold_accuraccy.append(accurracy_and_loss[1] * 100)
    each_fold_loss.append(accurracy_and_loss[0])

    fold_index += 1


print('#####################################################################')
print('Accuracy and loss for each fold')
for i,_ in enumerate(each_fold_accuraccy):
  print('______________________________________________')
  print(f'FOLD {i+1} ---> Loss:  {each_fold_loss[i]} , Accuracy:  {each_fold_accuraccy[i]}%')
print('#####################################################################')
print(f'Mean Accuracy {np.mean(each_fold_accuraccy)} , Mean Loss:  {each_fold_loss[i]}%')
print('#####################################################################')


# Ploting confusion matrix 
model.load_weights("Z:/Master I/PML - Practical Machine Learning/Kaggle_Project/checkpoints/cv_5/1/best_checkpoint.ckpt")
predicted_labels = model(train_data[-2000:])
predicted_labels = np.argmax(predicted_labels, 1).reshape(len(predicted_labels),1)
metrics.ConfusionMatrixDisplay.from_predictions(train_labels[-2000:], predicted_labels).plot()

