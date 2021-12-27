import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Get the keras CNN model 
def getModel():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    return model


#to get all the images to the directory
def getImageinDirectory(train):
    if(len(os.listdir('./source/'+train.name+'/fear'))):
        filelist = [f for f in os.listdir(
            './source/'+train.name+'/fear') if f.endswith(".png")]
        for f in filelist:
            os.remove(os.path.join('./source/'+train.name+'/fear', f))
    
    if(len(os.listdir('./source/'+train.name+'/happy'))):
        filelist = [f for f in os.listdir(
            './source/'+train.name+'/happy') if f.endswith(".png")]
        for f in filelist:
            os.remove(os.path.join('./source/'+train.name+'/happy', f))

    if(len(os.listdir('./source/'+train.name+'/sad'))):
        filelist = [f for f in os.listdir(
            './source/'+train.name+'/sad') if f.endswith(".png")]
        for f in filelist:
            os.remove(os.path.join('./source/'+train.name+'/sad', f))
            
    fear = 0
    happy = 0
    sad = 0
    mat = np.zeros((48, 48), dtype=np.uint8)
    for i in range(len(train)):
        words = train.iloc[i, 1:].values
        # the image size is 48x48
        for j in range(2304):
            xind = j // 48
            yind = j % 48
            mat[xind][yind] = words[j]

        img = Image.fromarray(mat)
        if train.iloc[i, 0] == 'Fear':
            img.save('./source/'+train.name+'/fear/fear'+str(fear)+'.png')
            fear += 1
        elif train.iloc[i, 0] == 'Happy':
            img.save('./source/'+train.name+'/happy/happy'+str(happy)+'.png')
            happy += 1
        elif train.iloc[i, 0] == 'Sad':
            img.save('./source/'+train.name+'/sad/sad'+str(sad)+'.png')
            sad += 1




'''
The following dummy code for demonstration.
'''
def train_a_model(traincsv):
    '''
    :param traincsv:
    :return:
    '''
    train, validation = train_test_split(traincsv, test_size=0.1111)
    train.name = 'train'
    validation.name = 'validation'
    getImageinDirectory(validation)
    getImageinDirectory(train)

    train_dir = './source/train'
    val_dir = './source/validation'
    num_train = len(train)
    num_val = len(validation)
    batch_size = 32
    num_epoch = 50
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')
    
    #Train th model
    model = getModel()
    model.compile(loss='categorical_crossentropy', optimizer=Adam(
        lr=0.0001, decay=1e-6), metrics=['accuracy'])
    model_info = model.fit_generator(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size)
    model.save_weights('./source/model.h5')


    # Plotting the graph for accuracy and losses 
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(range(1, len(model_info.history['accuracy'])+1),
                model_info.history['accuracy'])
    axs[0].plot(range(1, len(model_info.history['val_accuracy'])+1),
                model_info.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_info.history['accuracy'])+1),
                    len(model_info.history['accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    axs[1].plot(range(1, len(model_info.history['loss'])+1),
                model_info.history['loss'])
    axs[1].plot(range(1, len(model_info.history['val_loss'])+1),
                model_info.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_info.history['loss'])+1),
                    len(model_info.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('./source/accuracy_and_loss_graphs.png')
    plt.show()
    pass


def test_the_model(test):
    '''
    :param testfile:
    :return:  a list of predicted values in same order of
    '''
    test.name = 'test'
    getImageinDirectory(test)
    model = getModel()
    model.load_weights('./source/model.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    mood_dict = {0: "Fear", 1: "Happy", 2: "Sad"}
    test_dir = './source/test'
    test_val = len(test)
    batch_size = 1

    # Predict the model
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(48,48),
            batch_size=batch_size,
            color_mode="grayscale",
            shuffle = False,
            class_mode='categorical')

    filenames = test_generator.filenames
    nb_samples = len(filenames)
    predict = model.predict_generator(test_generator,steps = nb_samples)
    prediction = []

    # Predict the mood for the testing images
    for i in range(len(predict)):
        prediction.append(int(np.argmax(predict[i])))
        frame = cv2.imread('.\\source\\test\\'+filenames[i])
        facecasc = cv2.CascadeClassifier(
            './source/haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-10), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(
                cv2.resize(roi_gray, (48, 48)), -1), 0)
            maxindex = int(np.argmax(predict[i]))
            cv2.putText(frame, mood_dict[maxindex], (x, y+w),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imwrite('./source/output/' + filenames[i] + '.png', frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #plot  the heatmap
    test_labels = test_generator.classes
    cm = confusion_matrix(prediction, test_labels)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, fmt='g')  # annot=True to annotate cells
    # labels, title and ticks
    ax.set_xlabel('Predicted Moods')
    ax.set_ylabel('True Moods')
    ax.set_title('Confusion Matrix of with accuracy of %s' %
                 (str(round(accuracy_score(prediction, test_labels),2))))
    ax.xaxis.set_ticklabels(['fear', 'happy', 'sad'])
    ax.yaxis.set_ticklabels(['fear', 'happy', 'sad'])
    
    plt.savefig('./source/confusion-matrix.png')
    # map the predicted mood to the mood_dict
    res = [mood_dict[i] for i in prediction]
    # return expected value
    return res 
