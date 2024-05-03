import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten

def load_model() -> Sequential:
    # Create a Sequential model
    model = Sequential()
    
    # Add a 3D convolutional layer with 128 filters, kernel size of 3, and input shape of (75, 46, 140, 1)
    model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
    model.add(Activation('relu'))  # Add a ReLU activation function
    model.add(MaxPool3D((1, 2, 2)))  # Add a 3D max pooling layer
    
    # Add another 3D convolutional layer with 256 filters and kernel size of 3
    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))  # Add a ReLU activation function
    model.add(MaxPool3D((1, 2, 2)))  # Add a 3D max pooling layer
    
    # Add another 3D convolutional layer with 75 filters and kernel size of 3
    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))  # Add a ReLU activation function
    model.add(MaxPool3D((1, 2, 2)))  # Add a 3D max pooling layer
    
    # Add a TimeDistributed layer to apply Flatten to each time step
    model.add(TimeDistributed(Flatten()))
    
    # Add a Bidirectional LSTM layer with 128 units and orthogonal kernel initializer, returning sequences
    model.add(Bidirectional(LSTM(128, kernel_initializer='orthogonal', return_sequences=True)))
    model.add(Dropout(.5))  # Add a dropout layer with rate of 0.5
    
    # Add another Bidirectional LSTM layer with 128 units and orthogonal kernel initializer, returning sequences
    model.add(Bidirectional(LSTM(128, kernel_initializer='orthogonal', return_sequences=True)))
    model.add(Dropout(.5))  # Add another dropout layer with rate of 0.5
    
    # Add a Dense layer with 41 units, He normal kernel initializer, and softmax activation
    model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))
    
    # Load weights from a checkpoint file
    model.load_weights('models/checkpoint')
    
    return model
