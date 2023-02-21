########################## TRIED MODEL ###############################
# * submission_Patrascu_Rares_2 fitted with last 2000 as validation test acc 0.90(SGD, 100 epochs, lr= 0.01)
# * submission_Patrascu_Rares_3 fitted on all data test acc 0.92 (SGD, 50 epochs, lr= 0.01)


model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),input_shape = (150,3,1), activation='linear', strides=(1,1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='linear', strides=(1,1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='linear', strides=(1,1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
        tf.keras.layers.Conv2D(filters=248, kernel_size=(3, 3), activation='linear', strides=(1,1), padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='linear', strides=(1,1), padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3), 
        tf.keras.layers.Dense(20, activation='softmax') ])

########################## TRIED MODEL ###############################
# * Added another 248 layer 
# * submission_ last 200 validation test acc 0.913(SGD, 100 epochs, lr= 0.01)
# * submission_ fitted on all data test acc 9.32(SGD, 50 epochs, lr= 0.01)

model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),input_shape = (150,3,1), activation='linear', strides=(1,1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='linear', strides=(1,1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='linear', strides=(1,1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
        tf.keras.layers.Conv2D(filters=248, kernel_size=(3, 3), activation='linear', strides=(1,1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
        tf.keras.layers.Conv2D(filters=248, kernel_size=(3, 3), activation='linear', strides=(1,1), padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.3), 
        tf.keras.layers.Dense(20, activation='softmax') 
            ])
    
########################## TRIED MODEL ###############################
# Added a GlobalAveragePooling2D layer and some more 
# * submission_Patrascu_Rares_13 trained with 2000 validatiom 0.94 test acc
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),input_shape = (150,3,1), activation='linear', strides=(1,1), padding='same'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
  tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='linear', strides=(1,1), padding='same'),
  tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='linear', strides=(1,1), padding='same'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
  tf.keras.layers.Conv2D(filters=248, kernel_size=(3, 3), activation='linear', strides=(1,1), padding='same'),
  tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Conv2D(filters=248, kernel_size=(3, 3), activation='linear', strides=(1,1), padding='same'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
  tf.keras.layers.Conv2D(filters=496, kernel_size=(3, 3), activation='linear', strides=(1,1), padding='same'),
  tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.GlobalAveragePooling2D(data_format=None, keepdims=False),
  tf.keras.layers.Dense(50, activation='linear'),
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Dense(20, activation='softmax')
])

########################## TRIED MODEL ###############################
 # * trained using cross fold validation with 5 folds and tried different setups for lr decay 
 # * best model was obtained with the second schedular used, SGD as optimizer and saving the best checkpoint for each fold based on val acc 
 # * submission_  test acc 9.37 
 # * To note the model was also trained using a droput after each batchnorm and got the same test acc however after the final standing where shown it got a better overall accuracy
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
########################## TRIED MODEL ###############################
# * After reading some papers i decided to train the model more times with Bathnorm, Droput, and activation ordered in each way 

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),input_shape = (150,3,1), activation=tf.keras.layers.LeakyReLU(alpha=0.01), strides=(1,1), padding='same'),
  tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01), strides=(1,1), padding='same'),
  tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01), strides=(1,1), padding='same'),
  tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Conv2D(filters=248, kernel_size=(3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01), strides=(1,1), padding='same'),
  tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Conv2D(filters=248, kernel_size=(3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01), strides=(1,1), padding='same'),
  tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Conv2D(filters=496, kernel_size=(3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01), strides=(1,1), padding='same'),
  tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Conv2D(filters=496, kernel_size=(3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.01), strides=(1,1), padding='same'),
  tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.GlobalAveragePooling2D(data_format=None, keepdims=False),
  tf.keras.layers.Dense(50, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(20, activation='softmax')
])

########################## TRIED MODEL ###############################
# * I Have tried a bigger model with a more powerful lr decay after the first rounds and also cganged the activation between Relu,Linear and LeakyRelu
# * submission_  test acc 9.34 
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),input_shape = (150,3,1), strides=(1,1), padding='same'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.01),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),strides=(1,1), padding='same'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.01),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),strides=(1,1), padding='same'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.01),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1), padding='same'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.01),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1,1), padding='same'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.01),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1,1), padding='same'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.01),
    tf.keras.layers.Conv2D(filters=248, kernel_size=(3, 3), strides=(1,1), padding='same'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.01),
    tf.keras.layers.Conv2D(filters=248, kernel_size=(3, 3), strides=(1,1), padding='same'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(alpha=0.01),
    tf.keras.layers.Conv2D(filters=496, kernel_size=(3, 3), strides=(1,1), padding='same'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.01),
    tf.keras.layers.Conv2D(filters=496, kernel_size=(3, 3), strides=(1,1), padding='same'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=(2,2), padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.01),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.GlobalAveragePooling2D(data_format=None, keepdims=False),
    tf.keras.layers.Dense(200, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(50, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(20, activation='softmax')
    ])
########################## TRIED MODEL ###############################

########################## TRIED MODEL ###############################

########################## TRIED MODEL ###############################

########################## TRIED MODEL ###############################