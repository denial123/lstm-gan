from __future__ import print_function, division
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob

from music21 import converter, instrument, note, chord, stream
from keras.layers import Input, Dense, MaxPooling1D, Reshape, Dropout, LSTM, Bidirectional, Conv1D, Flatten
from keras.models import Sequential, Model, model_from_json
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint

from numpy import asarray
from keras.layers import Dense
from keras.layers import Dropout
from random import shuffle

def train_network():
    """ Train a Neural Network to classify music """

    # Convert notes into numerical input
    dirs = ["love_simple/*.mid", "anger_simple/*.mid"]
    X_train, y_train = prepare_full_input(dirs)


    # Set up the model
    # (1/4).
    model = create_classifier(X_train, y_train)
    model.summary()
    # (2/4). hier #Epochen ändern
    n_epochs = 500
    #    #########
    # Create Checkpoint and Fit the model
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min',
        # (3/4). hier Häufigkeit der Gewichtespeicherung verändern
        period=30
    )
    callbacks_list = [checkpoint]
    # (4/4).
    history = model.fit(X_train, y_train, callbacks=callbacks_list, epochs=n_epochs, batch_size=64)
    # 5. ggf noch load weights einkommentieren
    #########


    # Plot the model losses
    pd.DataFrame(history.history).plot()
    plt.savefig('Classifier_Accuracy.png', transparent=True)
    plt.close()

    # Evaluate
    #gan_test, gan_label = prepare_gan_output(['cgan_final_0_2000_big_corpus.mid','cgan_final_1_2000_big_corpus.mid'], [0,1])
    gan_test, gan_label = prepare_gan_output(['outputs\love_midi_by_lstm_75_epochs.mid', 'outputs\\anger_midi_by_lstm_80_epochs.mid'], [0, 1])
    eval_loss, eval_acc = model.evaluate(gan_test,gan_label)
    print("Model accuracy: %.2f" %eval_acc)

    model_predictions = model.predict(gan_test)
    print(model_predictions)

    #for i in (0,gan_label):
     #   predicted_label = np.argmax(model_predictions[i])
     #   expected_label = gan_label[i]

    #print("Predicted: %f Expected: %f" % (predicted_label,expected_label))


""" method prepares two midi output files taken from GAN or CGAN to test the model"""
def prepare_gan_output(files, label_numbers):
    notes = []
    len_notes = []
    for file in files:
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None
        count = 0
        try:  # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:  # file has notes in a flat structure
            notes_to_parse = midi.flat.notes
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
                count+=1
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
                count+=1
        len_notes.append(count)
        print("len notes", len_notes)
    #Prepare the Sequences
    sequence_length = 80

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))
    print("pitchnames",pitchnames)

    # Create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    print(len(note_to_int))
    network_input = []

    for n in range (0, len(label_numbers)):
        for i in range(0, len_notes[n] - sequence_length,1):
            sequence_in = notes[i:i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])

    first_corpus_length = len_notes[0]-sequence_length
    while len(network_input)> 2*first_corpus_length:
        network_input.pop()
    n_patterns = len(network_input)
    print("n_patterns", n_patterns)

    label = []
    for i in range(0, len_notes[0]-sequence_length,1):
        label.append(label_numbers[0])
    print("len label",len(label))
    while len(label) < len(network_input):
        label.append((label_numbers[1]))
    print("len label", len(label))
    ##### NEW #####
    network_in_out = []
    # Prepare for shuffle
    for i in range(0, n_patterns):
        network_in_out.append((network_input[i], label[i]))
    # Shuffle
    shuffle(network_in_out)
    print(len(network_in_out))
    # Seperate lists
    network_input.clear()
    label.clear()
    for row in network_in_out:
        network_input.append(row[0])
        label.append(row[1])
    print("network input new", network_input)
    print("nework_input length", len(network_input))
    print("network output new", label)
    print("network output length", len(label))

    # n_patterns = len(network_input)
    # print("n patterns: ",n_patterns)
    ##### NEW end ####

    # Reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # Normalize input between 0 and 1
    network_input = network_input / float(len(notes))
    # Label as one-hot-encoded Vectors
    label = np_utils.to_categorical(label, num_classes=2)

    print("len", len(network_input))
    print("label", label)

    # Versuch ohne eckige Klammern
    return network_input, label


def create_classifier(X_train, y_train):
    """ create the structure of the neural network """
    model = Sequential()
    print(X_train.shape[1])
    print(X_train.shape[2])
    # add model layers
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape = (X_train.shape[1], X_train.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    #model.add(LSTM(512, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    #model.add(Dropout(0.3))
    #model.add(Bidirectional(LSTM(512, return_sequences=True)))
    #model.add(Dropout(0.3))
    #model.add(Bidirectional(LSTM(512)))
    #model.add(Dense(256, activation='sigmoid'))
    #model.add(Dropout(0.3))
    model.add(Dense(len(y_train[0]), activation='softmax'))
    # zum Gewichte laden einkommentieren
    #model.load_weights("weights-improvement-10-0.6526-bigger.hdf5")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("Created model and loaded weights from file")

    return model

def get_notes(dir="love_simple/*.mid"):
    """ Get all the notes and chords from the midi files """
    notes = []

    for file in glob.glob(dir):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try:  # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:  # file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))


    return notes

def prepare_sequences(notes, n_vocab, label_number=0):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 80

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))

    # Create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])

    n_patterns = len(network_input)

    # Reshape the input into a format compatible with LSTM layers
    #network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = asarray(network_input)
    # Normalize input between 0 and 1
    network_input = network_input / float(n_vocab)
    #network_output = np_utils.to_categorical(network_output)

    label = []
    for i in range(0, n_patterns):
        label.append(label_number)
    #print("network_input", network_input)
    print(len(label))
    print("label in prepare sequences",label)


    # Versuch ohne eckige Klammern
    return (network_input, label)

def prepare_full_input(dirs):
    dir_count = 0
    X_train = []
    y_train = []
    for dir in dirs:
        notes = get_notes(dir)
        n_vocab = len(set(notes))
        ### hier labels mit übergeben
        X_train_part, y_train_part = prepare_sequences(notes, n_vocab, dir_count)
        dir_count += 1
        for xt in X_train_part:
            X_train.append(xt)
        ### label einzeln anhängen
        for l in y_train_part:
            y_train.append(l)

    ##### NEW #####
    X_y_train = []
    # Prepare for shuffle
    for i in range(0, len(X_train)):
        X_y_train.append((X_train[i], y_train[i]))
    print("network in out before shuffle", X_y_train)
    print(len(X_y_train))
    # Shuffle
    shuffle(X_y_train)
    print("n i o after shuffle", X_y_train)
    print(len(X_y_train))
    # Seperate lists
    X_train.clear()
    y_train.clear()
    for row in X_y_train:
        X_train.append(row[0])
        y_train.append(row[1])
    #print("X train new", X_train)
    print("X train length", len(X_train))
    print("y train new", y_train)
    print("y train length", len(y_train))
    ##### NEW end ####

    # Reshape the input into a format compatible with LSTM layers
    sequence_length = 80
    X_train = np.reshape(X_train, (len(y_train), sequence_length, 1))
    print("X_train", X_train)
    print("len X train", len(X_train))
    y_train = np.reshape(y_train, (len(y_train)))
    y_train = np_utils.to_categorical(y_train)
    #print("y_train", y_train)
    #print("len y train", len(y_train))

    return X_train,y_train

if __name__ == '__main__':
   train_network()

