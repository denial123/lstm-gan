#pylab inline
import glob
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LSTM, Bidirectional
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, History


def train_network():
    """ Train a Neural Network to generate music """
    # Get notes from midi files
    notes = get_notes()

    # Get the number of pitch names
    n_vocab = len(set(notes))

    # Convert notes into numerical input
    network_input, network_output = prepare_sequences(notes, n_vocab)

    # Set up the model
    # (1/4). nutze: create_network 1 (aus lstm-gan mit BIDIRECTION) oder 2 (aus lstm ohne BIDIRECTION)
    model = create_network1(network_input, n_vocab)
    model.summary()
    # (2/4). hier #Epochen ändern
    n_epochs = 2
    #########
    # Create Checkpoint and Fit the model
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min',
        # (3/4). hier Häufigkeit der Gewichtespeicherung verändern
        period=2
    )
    callbacks_list = [checkpoint]
    # (4/4).wenn create_network1(): batch_size = 64, wenn create_network2(): batch_size = 128
    history = model.fit(network_input, network_output, callbacks=callbacks_list, epochs=n_epochs, batch_size=64)
    #5. ggf noch load weights einkommentieren
    #model.save('LSTMmodel.h5')
    #########

    # Use the model to generate a midi
    prediction_output = generate_notes(model, notes, network_input, len(set(notes)))
    create_midi(prediction_output, 'love_midi')
    
    # Plot the model losses
    pd.DataFrame(history.history).plot()
    plt.savefig('LSTM_Loss_per_Epoch.png', transparent=True)
    plt.close()
    
def get_notes():
    """ Get all the notes and chords from the midi files """
    notes = []

    for file in glob.glob("love_obpi/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try: # file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: # file has notes in a flat structure
            notes_to_parse = midi.flat.notes
            
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    print("len notes:", len(notes))
    print('len set notes: ', len(set(notes)))
    return notes
  
def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)
    print("n_patterns: ", n_patterns)

    # Reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

    print("network_input after reshape:", network_input)
    print("# columns: ", len(network_input[0]))
    print('# rows: ', len(network_input))

    # normalize input between 0 and 1
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)
    print('network_output', network_output)
    print("# columns: ", len(network_output[0]))
    print('# rows: ', len(network_output))
    return (network_input, network_output)
  
def create_network1(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(512, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(512)))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    # zum Gewichte laden einkommentieren
    model.load_weights("weights-improvement-02-3.7732-bigger.hdf5")
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print("Created model and loaded weights from file")

    return model

def create_network2(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,  # 512
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(256, return_sequences=True, recurrent_dropout=0.3, ))  # 512
    model.add(LSTM(512))  # 512
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(256))  # 256
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    # zum Gewichte laden einkommentieren
    # model.load_weights("weights-improvement-03-4.6675-bigger.hdf5")
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    print("Created model and loaded weights from file")
    return model

def generate_notes(model, notes, network_input, n_vocab):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    pitchnames = sorted(set(item for item in notes))
    
    start = np.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    print("pattern:", pattern)
    print("pattern len:", len(pattern))
    prediction_output = []

    # generate 500 notes
    for note_index in range(300): #500
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        
        pattern = np.append(pattern,index)
        pattern = pattern[1:len(pattern)]

    return prediction_output
  
def create_midi(prediction_output, filename):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.show('text')
    midi_stream.show()
    #einkommentieren, um midi file zu speichern
    #midi_stream.write('midi', fp='{}.mid'.format(filename))
    
if __name__ == '__main__':
   train_network()

