import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from glob import glob
from pathlib import Path
from itertools import chain
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from bidict import bidict
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense , GlobalAveragePooling2D , Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam , SGD
from tensorflow.keras.applications import VGG19 , ResNet101V2 , InceptionResNetV2, Xception , EfficientNetB4
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

selected_model = "InceptionResNetV2"
model_input_size={"VGG19":(224,224),"ResNet101V2":(224,224),"InceptionResNetV2":(299,299),"Xception":(299,299),"EfficientNetB4":(299,299)}
model_input_shape={"VGG19":(224,224,3),"ResNet101V2":(224,224,3),"InceptionResNetV2":(299,299,3),"Xception":(299,299,3),"EfficientNetB4":(299,299,3)}
MAX_CASES = 10000
MIN_CASES = 2000
BATCH_SIZE = 16
EPOCH = 50
LEARNING_RATE = 0.0001
CALLBACKS = [EarlyStopping(patience=5, monitor = "val_accuracy")]

dataset_path = Path("D:/AIMS/HW2").as_posix()
label_dict = bidict()

def create_dataframe():
    filename = Path(dataset_path).joinpath("Data_Entry_2017.csv").as_posix()
    data_frame = pd.read_csv(filename, usecols=["Image Index", "Finding Labels"])
    raw_paths = glob(os.path.join(dataset_path,"images*","*","*.png") )
    all_image_paths = { os.path.basename(i) : Path(i).as_posix() for i in  raw_paths}
    print("Scans found:"  ,len(all_image_paths)  ,"  ,Total Headers"  ,data_frame.shape[0])
    data_frame["path"] = data_frame["Image Index"].map(all_image_paths.get)
    # data_frame["Finding Labels"] = data_frame["Finding Labels"].map(lambda x: x.replace("No Finding"  ,""))
    all_labels = np.unique(list(chain(*data_frame["Finding Labels"].map(lambda x: x.split("|")).tolist() )))
    all_labels = [i for i in all_labels if len(i)>0]
    # 把多種疾病的資料去掉
    for i in range(len(all_labels)):
        label_dict[i] = str(all_labels[i])
    print("all_labels : ",len(all_labels))
    print(label_dict)
    index = []
    for idx , row in data_frame.iterrows():
        if "|" in row["Finding Labels"]:
            index.append(idx)
        # if "No Finding" in row["Finding Labels"]:
        #     index.append(idx)

    data_frame = data_frame.drop(data_frame.index[index])
    # -----------------------
    train_labels = data_frame.groupby(["Finding Labels"])["Finding Labels"].count()
    print("train_labels : ",train_labels)
    # -----------------------
    sample_df = data_frame.sample(5)
    # print(sample_df)
    # resample
    sample_weights = data_frame["Finding Labels"].map(lambda x: len(x.split("|")) if len(x)>0 else 0).values + 5e-2
    sample_weights /= sample_weights.sum()
    data_frame = data_frame.sample(20000 , weights=sample_weights) # 從所有資料中挑50000筆
    return data_frame , all_labels

def create_dataset(train_df , all_labels):
    # generator = ImageDataGenerator(samplewise_center=True,samplewise_std_normalization=True,horizontal_flip=True,height_shift_range= 0.05,width_shift_range=0.1,rotation_range=3,fill_mode="reflect",validation_split=0.2,zoom_range=0.1)
    generator = ImageDataGenerator(samplewise_center=True,samplewise_std_normalization=True,horizontal_flip=True,validation_split=0.2)
    train_batches = generator.flow_from_dataframe(dataframe=train_df,directory=dataset_path,x_col="path",y_col="Finding Labels",class_mode="categorical",target_size=model_input_size[selected_model],subset="training",shuffle=True,seed=30,batch_size=BATCH_SIZE)
    validation_batches = generator.flow_from_dataframe(dataframe=train_df,directory=dataset_path,x_col="path",y_col="Finding Labels",class_mode="categorical",target_size=model_input_size[selected_model],subset="validation",seed=30,batch_size=BATCH_SIZE)
    print("\n----------------------------------------SANITY CHECK LINE----------------------------------------")
    return train_batches , validation_batches

def build_model(num_classes):
    if selected_model == "VGG19":
        backbone = VGG19(include_top = False , weights = "imagenet" , input_shape = model_input_shape[selected_model])
    if selected_model == "ResNet101V2":
        backbone = ResNet101V2(include_top = False , weights = "imagenet" , input_shape = model_input_shape[selected_model])
    elif selected_model == "InceptionResNetV2":
        backbone = InceptionResNetV2(include_top = False , weights="imagenet" , input_shape = model_input_shape[selected_model])
    elif selected_model == "Xception":
        backbone = Xception(include_top = False , weights="imagenet" , input_shape = model_input_shape[selected_model])
    elif selected_model == "EfficientNetB4":
        backbone = EfficientNetB4(include_top = False , weights="imagenet" , input_shape = model_input_shape[selected_model])
    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(2048, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))
    selected_optimizer = Adam(learning_rate=LEARNING_RATE)
    # selected_optimizer = SGD(learning_rate=LEARNING_RATE , decay=1e-6 , momentum=0.9)
    model.compile(optimizer=selected_optimizer , loss="categorical_crossentropy" , metrics=["accuracy"])
    model.summary()
    return model

def train(model , train_batches , validation_batches , all_labels):
    history = model.fit(train_batches , steps_per_epoch=train_batches.samples//BATCH_SIZE , validation_data=validation_batches , validation_steps=validation_batches.samples//BATCH_SIZE , epochs=EPOCH , callbacks=CALLBACKS)
    model.save( Path(dataset_path).joinpath(selected_model+".h5").as_posix() )
    return history

#印出accuracy跟loss
def plot_training_history(history):
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(["train", "validation"], loc="lower right")
    plt.show()
    # loss graph
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["train", "validation"], loc="upper right")
    plt.show()

if __name__ == "__main__":
    train_df , all_labels= create_dataframe()
    train_batches , validation_batches = create_dataset(train_df , all_labels)
    model = build_model(len(all_labels))
    history = train(model , train_batches , validation_batches , all_labels)
    plot_training_history(history)