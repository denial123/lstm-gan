#Original Source: https://github.com/olofmogren/c-rnn-gan
try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse
from urllib.request import urlopen
import os, midi, math, random, re, string, sys
import numpy as np
from io import BytesIO

GENRE = 0
COMPOSER = 1
SONG_DATA = 2

# INDICES IN BATCHES (LENGTH,FREQ,VELOCITY are repeated self.tones_per_cell times):
TICKS_FROM_PREV_START = 0
LENGTH = 1
FREQ = 2
VELOCITY = 3

# INDICES IN SONG DATA (NOT YET BATCHED):
BEGIN_TICK = 0

NUM_FEATURES_PER_TONE = 3

debug = ''


class MusicDataLoader(object):

    def __init__(self, datadir, select_validation_percentage, select_test_percentage, works_per_composer=None,
                 pace_events=False, synthetic=None, tones_per_cell=1, single_composer=None):
        self.datadir = datadir
        self.output_ticks_per_quarter_note = 384.0
        self.tones_per_cell = tones_per_cell
        self.single_composer = single_composer
        self.pointer = {}
        self.pointer['validation'] = 0
        self.pointer['test'] = 0
        self.pointer['train'] = 0
        if synthetic == 'chords':
            self.generate_chords(pace_events=pace_events)
        elif not datadir is None:
            print('Data loader: datadir: {}'.format(datadir))
            self.download_midi_data()
            self.read_data(select_validation_percentage, select_test_percentage, works_per_composer, pace_events)

    def read_one_file(self, filename, pace_events):
        try:
            # midi_pattern = midi.read_midifile(os.path.join(path, filename))
            midi_pattern = midi.read_midifile(filename)
            print(('Reading {}'.format(filename)))
            print("Tracks in midi_pattern", len(midi_pattern))
        except:
            print('Error reading {}'.format(filename))
            return None
        song_data = []

        # Tempo:
        ticks_per_quarter_note = float(midi_pattern.resolution)
        input_ticks_per_output_tick = ticks_per_quarter_note / self.output_ticks_per_quarter_note
        # if debug == 'overfit': input_ticks_per_output_tick = 1.0

        # Multiply with output_ticks_pr_input_tick for output ticks.
        # --- only melody track = first track
        track = midi_pattern[0]
        last_event_input_tick = 0
        not_closed_notes = []
        for event in track:
            if type(event) == midi.events.SetTempoEvent:
                pass  # These are currently ignored
            elif (type(event) == midi.events.NoteOffEvent) or \
                    (type(event) == midi.events.NoteOnEvent and \
                     event.velocity == 0):
                retained_not_closed_notes = []
                for e in not_closed_notes:
                    if tone_to_freq(event.data[0]) == e[FREQ]:
                        event_abs_tick = float(event.tick + last_event_input_tick) / input_ticks_per_output_tick
                        e[LENGTH] = event_abs_tick - e[BEGIN_TICK]
                        song_data.append(e)
                    else:
                        retained_not_closed_notes.append(e)
                not_closed_notes = retained_not_closed_notes
            elif type(event) == midi.events.NoteOnEvent:
                begin_tick = float(event.tick + last_event_input_tick) / input_ticks_per_output_tick
                note = [0.0] * (NUM_FEATURES_PER_TONE + 1)
                note[FREQ] = tone_to_freq(event.data[0])
                note[VELOCITY] = float(event.data[1])
                note[BEGIN_TICK] = begin_tick
                not_closed_notes.append(note)
            last_event_input_tick += event.tick
        for e in not_closed_notes:
            # print (('Warning: found no NoteOffEvent for this note. Will close it. {}'.format(e))
            e[LENGTH] = float(ticks_per_quarter_note) / input_ticks_per_output_tick
            song_data.append(e)
        song_data.sort(key=lambda e: e[BEGIN_TICK])
        #Tempo Events
        if (pace_events):
            pace_event_list = []
            pace_tick = 0.0
            song_tick_length = song_data[-1][BEGIN_TICK] + song_data[-1][LENGTH]
            while pace_tick < song_tick_length:
                song_data.append([0.0, 440.0, 0.0, pace_tick, 0.0])
                pace_tick += float(ticks_per_quarter_note) / input_ticks_per_output_tick
            song_data.sort(key=lambda e: e[BEGIN_TICK])
        return song_data

    def get_midi_pattern(self, song_data):
        """
        get_midi_pattern takes a song in internal representation
        (a tensor of dimensions [songlength, self.num_song_features]).
        the three values are length, frequency, velocity.
        if velocity of a frame is zero, no midi event will be
        triggered at that frame.
        returns the midi_pattern.
        Can be used with filename == None. Then nothing is saved, but only returned.
        """

        # Tempo:
        # Multiply with output_ticks_pr_input_tick for output ticks.
        midi_pattern = midi.Pattern([], resolution=int(self.output_ticks_per_quarter_note))
        cur_track = midi.Track([])
        cur_track.append(midi.events.SetTempoEvent(tick=0, bpm=45))
        future_events = {}
        last_event_tick = 0

        ticks_to_this_tone = 0.0
        song_events_absolute_ticks = []
        abs_tick_note_beginning = 0.0
        for frame in song_data:
            abs_tick_note_beginning += frame[TICKS_FROM_PREV_START]
            for subframe in range(self.tones_per_cell):
                offset = subframe * NUM_FEATURES_PER_TONE
                tick_len = int(round(frame[offset + LENGTH]))
                freq = frame[offset + FREQ]
                velocity = min(int(round(frame[offset + VELOCITY])), 127)
                # print (('tick_len: {}, freq: {}, velocity: {}, ticks_from_prev_start: {}'.format(tick_len, freq, velocity, frame[TICKS_FROM_PREV_START]))
                d = freq_to_tone(freq)
                # print (('d: {}'.format(d))
                if d is not None and velocity > 0 and tick_len > 0:
                    # range-check with preserved tone, changed one octave:
                    tone = d['tone']
                    while tone < 0:   tone += 12
                    while tone > 127: tone -= 12
                    pitch_wheel = cents_to_pitchwheel_units(d['cents'])
                    # print (('tick_len: {}, freq: {}, tone: {}, pitch_wheel: {}, velocity: {}'.format(tick_len, freq, tone, pitch_wheel, velocity))
                    # if pitch_wheel != 0:
                    # midi.events.PitchWheelEvent(tick=int(ticks_to_this_tone),
                    #                                            pitch=pitch_wheel)
                    song_events_absolute_ticks.append((abs_tick_note_beginning,
                                                       midi.events.NoteOnEvent(
                                                           tick=0,
                                                           velocity=velocity,
                                                           pitch=tone)))
                    song_events_absolute_ticks.append((abs_tick_note_beginning + tick_len,
                                                       midi.events.NoteOffEvent(
                                                           tick=0,
                                                           velocity=0,
                                                           pitch=tone)))
        song_events_absolute_ticks.sort(key=lambda e: e[0])
        abs_tick_note_beginning = 0.0
        for abs_tick, event in song_events_absolute_ticks:
            rel_tick = abs_tick - abs_tick_note_beginning
            event.tick = int(round(rel_tick))
            cur_track.append(event)
            abs_tick_note_beginning = abs_tick

        cur_track.append(midi.EndOfTrackEvent(tick=int(self.output_ticks_per_quarter_note)))
        midi_pattern.append(cur_track)
        return midi_pattern

    def save_midi_pattern(self, filename, midi_pattern):
        if filename is not None:
            midi.write_midifile(filename, midi_pattern)

    def save_data(self, filename, song_data):
        """
        save_data takes a filename and a song in internal representation
        (a tensor of dimensions [songlength, 3]).
        the three values are length, frequency, velocity.
        if velocity of a frame is zero, no midi event will be
        triggered at that frame.
        returns the midi_pattern.
        Can be used with filename == None. Then nothing is saved, but only returned.
        """
        midi_pattern = self.get_midi_pattern(song_data)
        self.save_midi_pattern(filename, midi_pattern)
        return midi_pattern


def tone_to_freq(tone):
    # returns the frequency of a tone.
    # formulas from
    # * https://en.wikipedia.org/wiki/MIDI_Tuning_Standard
    # * https://en.wikipedia.org/wiki/Cent_(music)

    return math.pow(2, ((float(tone) - 69.0) / 12.0)) * 440.0


def freq_to_tone(freq):
    """
    returns a dict d where
    d['tone'] is the base tone in midi standard
    d['cents'] is the cents to make the tone into the exact-ish frequency provided.
               multiply this with 8192 to get the midi pitch level.
    formulas from
      * https://en.wikipedia.org/wiki/MIDI_Tuning_Standard
      * https://en.wikipedia.org/wiki/Cent_(music)
  """
    if freq <= 0.0:
        return None
    float_tone = (69.0 + 12 * math.log(float(freq) / 440.0, 2))
    int_tone = int(float_tone)
    cents = int(1200 * math.log(float(freq) / tone_to_freq(int_tone), 2))
    return {'tone': int_tone, 'cents': cents}


def cents_to_pitchwheel_units(cents):
    return int(40.96 * (float(cents)))


def onehot(i, length):
    a = np.zeros(shape=[length])
    a[i] = 1
    return a


if __name__ == '__main__':
    # filename = sys.argv[1]
    filename = 'love_obpi/All_Out_Of_Love_obpi_trans_short.mid'
    print(('File: {}'.format(filename)))
    dl = MusicDataLoader(datadir=None, select_validation_percentage=0.0, select_test_percentage=0.0)
    abs_song_data = dl.read_one_file(filename=filename, pace_events=True)

    rel_song_data = []
    last_start = None
    for i, e in enumerate(abs_song_data):
        this_start = e[3]
        if last_start:
            e[3] = e[3] - last_start
        rel_song_data.append(e)
        last_start = this_start
    if not os.path.exists("newest_midi.mid"):
            print(('Saving: {}.'.format("new_midi.mid")))

            # Print midi pattern
            print(dl.save_data("new_midi.mid", rel_song_data))
    else:
            print(('File already exists: {}. Not saving.'.format("new_midi")))



