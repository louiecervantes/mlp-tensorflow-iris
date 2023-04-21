#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation



# Define the Streamlit app
def app():
    
    st.title('MLP Neural Network Using Tensorflow and Keras on the Iris dataset')
    st.subheader('by Louie F. Cervantes M.Eng., WVSU College of ICT')
    st.subheader('The Iris Dataset')

    st.write('The Iris dataset is a widely used benchmark dataset in machine learning \
    and statistics. It is a multivariate dataset that contains measurements for the \
    sepal length, sepal width, petal length, and petal width of 150 iris flowers, \
    belonging to three different species: Iris setosa, Iris versicolor, and Iris virginica.')

    st.write('The dataset was first introduced by the statistician and biologist Ronald \
    Fisher in 1936, and it has since become a classic example of exploratory data \
    analysis and classification. The dataset is often used to demonstrate classification \
    algorithms and techniques in machine learning, as the goal is to predict the \
    species of the iris flower based on the measurements of its sepal and petal dimensions.')
    st.subheader('Load the Iris Dataset')
    
    # Load the Iris dataset
    df = load_iris()
    data = pd.DataFrame(df.data)
    data.columns = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    st.dataframe(data, use_container_width=True)  
    
    st.subheader('Configuring the Neural Net')
    with st.echo(code_location='below'):
        #set the number of hidden layers
        neurons = st.slider('No. of neuron in the hidden layer', 5, 15, 10)
        #set the number or iterations
        epochs = st.slider('Number of epochs', 50, 250, 100, 10)
        if st.button('Run the Classifier'):
            # Split the dataset into training and testing sets
            x_train, x_test, y_train, y_test = train_test_split(df.data, df.target, test_size=0.2, random_state=42)

            # Normalize the input features
            mean = x_train.mean(axis=0)
            std = x_train.std(axis=0)
            x_train = (x_train - mean) / std
            x_test = (x_test - mean) / std

            # Define the model architecture
            model = tf.keras.models.Sequential([
              tf.keras.layers.Dense(neurons, activation='relu', input_shape=(4,)),
              tf.keras.layers.Dense(3, activation='softmax')
            ])

            # Compile the model with an optimizer, loss function, and metric
            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])

            # define a callback to write the output of each epoch to a text file
            class EpochCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs={}):
                    st.write(f"Epoch {epoch+1}: loss={logs['loss']:.4f}, acc={logs['accuracy']:.4f}\n")

            callback = EpochCallback()
            
            # Train the model on the training set
            history = model.fit(x_train, y_train, epochs=epochs, \
                                validation_data=(x_test, y_test), callbacks=[callback])

            # Evaluate the model on the test set
            test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
            st.write('Test accuracy:' + str(test_acc))
            
    st.write('In this version of the MLP we used the Keras library running on Tensorflow.  \
            Keras is a high-level neural network library written in Python that can run \
            on top of TensorFlow, Theano, and other machine learning frameworks. \
            It was developed to make deep learning more accessible and easy to use \
            for researchers and developers.  TensorFlow provides a platform for \
            building and deploying machine learning models. It is designed to \
            be scalable and can be used to build models ranging from small experiments\
            to large-scale production systems. TensorFlow supports a wide range of \
            machine learning algorithms, including deep learning, linear regression, \
            logistic regression, decision trees, and many others.')        
   
#run the app
if __name__ == "__main__":
    app()
