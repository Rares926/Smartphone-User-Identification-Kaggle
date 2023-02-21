import tensorflow as tf
import numpy as np 
import pandas as pd 
import os 
from tensorflow.keras.optimizers import SGD, Adam


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

def load_data_flatten(data_path: str, column_size: int= 150):
    """
    Loads the training data in a flatten numpy object. 
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

        return np.array(tmp_testing_re).T.flatten()[:column_size*3]

    for file in files: 
        print(file)
        one_input = pd.read_csv(file, header=None, names=['x', 'y', 'z']).values
        print(one_input.shape)
        all_lenghts.append(len(one_input))
        one_input = preproc_input(one_input)
        test_data.append(np.array(one_input))
        # print(np.mean(all_lenghts))
    print(np.mean(all_lenghts))
    return np.array(test_data)


test_data =  load_data_flatten("dataset/test/",156)
train_data = load_data_flatten("dataset/train/",156)
train_labels = load_labels("dataset/train_labels.csv")

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(512, activation='relu'), 
  tf.keras.layers.Dense(512, activation='relu'), 
  tf.keras.layers.Dropout(0.2), 
  tf.keras.layers.Dense(1024, activation='relu'), 
  tf.keras.layers.Dense(1024, activation='relu'), 
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(20, activation='softmax') 
])

optimizer = SGD(learning_rate=0.001, momentum=0.9)  

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data[:-2000], train_labels[:-2000],
          epochs=50, batch_size=32, initial_epoch=0,
          validation_data=(train_data[-2000:], train_labels[-2000:])) 