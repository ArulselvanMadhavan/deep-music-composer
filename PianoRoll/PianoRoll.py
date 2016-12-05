"""
Author: Arulselvan Madhavan
github: https://github.com/Arulselvanmadhavan
"""
import numpy as np
from mido import MidiFile, MidiTrack, Message
from mido import MetaMessage
from functools import partial
import warnings
import glob
from Utils import Utils
import Constants
DEFAULT_TICKS_PER_BEAT = 96
from music21 import *
from itertools import groupby


class PianoRoll(object):
    """Piano Roll Representation."""

    def __init__(self, path_to_data, res_factor=12):
        Utils.validate_path(path_to_data)
        self.path = path_to_data
        self.res_factor = res_factor
        self.files = Utils.get_matching_files(self.path+"*."+ Constants.MIDI_FILE_EXTENSION)
        self.ticks, self.max_note, self.min_note = PianoRoll.collect_stats(self.files, self.res_factor)

    @staticmethod
    def chord2roll(chord_file, num_ticks, lowest_note, num_cols, res_factor):
        cstream = converter.parse(chord_file)
        part = cstream.elements[0]
        timesteps_so_far = 0
        ticks_per_beat = DEFAULT_TICKS_PER_BEAT / res_factor
        roll = np.zeros((num_ticks, num_cols))
        for nr in part.notesAndRests:
            if isinstance(nr, note.Rest):
                c_pitches = [0]
            else:
                c_pitches = [p.midi - lowest_note + 1 for p in nr.pitches]
            start = timesteps_so_far
            dur_count = float(nr.quarterLength)
            timesteps_so_far += int(ticks_per_beat * dur_count)
            end = timesteps_so_far
            roll[start:end, c_pitches] = 1.0
        return roll

    @staticmethod
    def chord2roll_wrapper(chord_dir, num_ticks, lowest_note, num_cols, res_factor):
        chord_files = glob.glob("%s*.mid" % (chord_dir))
        piano_roll = np.zeros((len(chord_files), num_ticks, num_cols))
        for chord_file_id in range(len(chord_files)):
            roll = PianoRoll.chord2roll(chord_files[chord_file_id], num_ticks, lowest_note, num_cols, res_factor)
            piano_roll[chord_file_id, :, :] = roll
        return piano_roll

    @staticmethod
    def melody2roll_wrapper(midi_dir, num_ticks, lowest_note, num_cols, keys_count={}, res_factor=12):
        midi_files = Utils.get_matching_files(midi_dir + "*.mid")
        piano_roll = np.zeros((len(midi_files), num_ticks, num_cols))
        for i in range(len(midi_files)):
            midi_file = midi_files[i]
            piano_roll[i, :, :], keys_count = PianoRoll.melody2roll(midi_file,
                                                        num_ticks,
                                                        lowest_note,
                                                        num_cols,
                                                        keys_count,
                                                        res_factor)
        return piano_roll, keys_count

    @staticmethod
    def melody2roll(path_to_file,
                    num_ticks,
                    lowest_note,
                    num_cols,
                    keys_count,
                    res_factor = 12):
        ticks_per_beat = DEFAULT_TICKS_PER_BEAT / res_factor  # Integer division
        midi_stream = converter.parse(path_to_file)
        roll = np.zeros((num_ticks, num_cols))
        timesteps_so_far = 0
        for nr in midi_stream.elements[0].notesAndRests:
            p = nr.pitch.midi if isinstance(nr, note.Note) else 0
            duration = float(nr.quarterLength)
            col_idx = (
                      p - lowest_note) + 1 if p != 0 else p  # Adding to distinguish the case where there is no note.
            start = timesteps_so_far
            timesteps_so_far += int(ticks_per_beat * duration)
            end = timesteps_so_far
            roll[start:end, col_idx] = 1.0
        max_pitches = np.argmax(roll, axis=1)
        max_pitches[max_pitches != 0] += lowest_note
        for key, group in groupby(max_pitches):
            keys_count[key] = keys_count.get(key, 0) + len(list(group))
        return roll, keys_count

    @staticmethod
    def roll2midi_wrapper(composition_file, roll, lowest_note, res_factor, keys_count):
        quarter_note_resolution = 1. / (DEFAULT_TICKS_PER_BEAT / res_factor)
        max_pitches = np.argmax(roll, axis=1)
        max_pitches[max_pitches != 0] += lowest_note
        pitch_with_duration = [(key, len(list(group))) for key, group in groupby(max_pitches)]
        for key, group in groupby(max_pitches):
            keys_count[key] = keys_count.get(key, 0) + len(list(group))
        s = stream.Stream()
        for i in range(0, len(pitch_with_duration)):
            pitch_val, dur_count = pitch_with_duration[i]
            if (pitch_val == 0):
                n = note.Rest()
                n.duration.quarterLength = (quarter_note_resolution * dur_count)
            else:
                p = pitch.Pitch()
                p.midi = pitch_val
                n = note.Note(p.midi, quarterLength=(quarter_note_resolution * dur_count))
            s.append(n)
        score = stream.Stream()
        part = stream.Part()
        # part.clef = clef.BassClef()
        # part.append(instrument.Harpsichord())
        part.insert(s)
        score.insert(part)
        mf = midi.translate.streamToMidiFile(score)
        mf.open(composition_file, 'wb')
        mf.write()
        mf.close()
        return keys_count

    @staticmethod
    def message_stats(result, msg, res_factor):
        if not isinstance(msg, MetaMessage):
            num_ticks = result[0] + int(msg.time/res_factor)
            max_note = max(result[1], msg.note)
            min_note = min(result[2], msg.note)
            result = (num_ticks, max_note, min_note)
        return result

    @staticmethod
    def track_stats(result, track, res_factor):
        msg_stats = reduce(partial(PianoRoll.message_stats, res_factor=res_factor),
                           track,
                           (0, result[1], result[2]))
        return (max(result[0], msg_stats[0]), msg_stats[1], msg_stats[2])

    @staticmethod
    def midi_stats(result, file, res_factor):
        mid = MidiFile(file)
        return reduce(partial(PianoRoll.track_stats, res_factor=res_factor), mid.tracks, result)

    @staticmethod
    def collect_stats(files, res_factor):
        stats = reduce(partial(PianoRoll.midi_stats, res_factor=res_factor),
                       files,
                       (Constants.MAX_TICK_INITIALIZER, Constants.MAX_NOTE_INITIALIZER, Constants.MIN_NOTE_INITIALIZER))
        return stats[0], stats[1], stats[2]
    #
    # @staticmethod
    # def get_note_on_off_info(track, configs):
    #     result = reduce(partial(PianoRoll.get_note_length, configs=configs),
    #                     track,
    #                     (0, [], [-1] * (configs[Constants.MAX_NOTE] - configs[Constants.MIN_NOTE] + 1)))
    #     return result[1]
    #
    # @staticmethod
    # def parse_midi_message(result, message, configs):
    #     index, current_time, notes = result
    #     if not isinstance(message, MetaMessage):
    #         current_time += int(message.time/configs[Constants.RES_FACTOR])
    #         if message.type == Constants.NOTE_ON:
    #             note_onoff = 1
    #         elif message.type == Constants.NOTE_OFF:
    #             note_onoff = 0
    #         else:
    #             print("Error: Note Type not recognized!")
    #         notes[index] = [message.note, current_time, note_onoff]
    #         index += 1
    #     return (index, current_time, notes)
    #
    # @staticmethod
    # def get_note_length(result, msg, configs):
    #     """
    #     :param result: (current_time, notes, start_time_tracker)
    #     :param msg: MidiMessage
    #     :param configs:
    #     :return:
    #     """
    #     current_time, notes, start_time_tracker = result
    #     if not isinstance(msg, MetaMessage):
    #         resolved_time = int(msg.time/configs[Constants.RES_FACTOR])
    #         current_time += resolved_time
    #         if msg.type == Constants.NOTE_ON:
    #             start_time_tracker[msg.note - configs[Constants.MIN_NOTE]] = current_time
    #         elif msg.type == Constants.NOTE_OFF:
    #             start_time = start_time_tracker[msg.note - configs[Constants.MIN_NOTE]]
    #             if start_time == -1:
    #                 warnings.warn("End note occurs without a start")
    #             notes.append([msg.note, start_time, current_time - start_time])
    #     return (current_time, notes, start_time_tracker)
    #
    # @staticmethod
    # def enter_piano_notes(piano_roll, msg, min_note):
    #     piano_roll[0, msg[1]:(msg[1]+int(msg[2]/2)), msg[0]-min_note] = 1
    #     return piano_roll
    #
    # @staticmethod
    # def get_note_info(file_path, configs):
    #     mid = MidiFile(file_path)
    #     track_notes = map(partial(PianoRoll.get_note_on_off_info, configs=configs),
    #                       mid.tracks)
    #     notes_per_file = np.vstack(track_notes)
    #     if len(track_notes) > 1 : #TODO - DEBUG Block Remove later.
    #         print("More than 1 track in {}".format(file_path))
    #     piano_roll_per_file = np.zeros((1, configs[Constants.TICKS],
    #                                     configs[Constants.MAX_NOTE]-configs[Constants.MIN_NOTE]+1), dtype=np.float32)
    #     return reduce(partial(PianoRoll.enter_piano_notes, min_note=configs[Constants.MIN_NOTE]),
    #                   notes_per_file,
    #                   piano_roll_per_file)
    #
    # def generate_piano_roll_func(self):
    #     configs = {
    #         Constants.TICKS : self.ticks,
    #         Constants.RES_FACTOR : self.res_factor,
    #         Constants.MAX_NOTE : self.max_note,
    #         Constants.MIN_NOTE : self.min_note
    #     }
    #     final = map(partial(PianoRoll.get_note_info, configs=configs),
    #                                 self.files)
    #     return np.vstack(final)
    #
    # @staticmethod #TODO - Make this functional
    # def generate_samples(chord_roll, melody_roll, seq_length):
    #     chord_roll = np.tile(chord_roll, (1, 2, 1))
    #     melody_roll = np.tile(melody_roll, (1, 2, 1))
    #     X = []
    #     y = []
    #     for i, song in enumerate(chord_roll):
    #         pos = 0
    #         while pos+seq_length < song.shape[0]:
    #             sequence = np.array(song[pos:pos+seq_length])
    #             X.append(sequence)
    #             y.append(melody_roll[i, pos+seq_length])
    #             pos += 1
    #     return np.array(X), np.array(y)
    #
    # @staticmethod
    # def generate_test_samples(chord_roll, seq_length):
    #     chord_roll = np.tile(chord_roll, (1, 2, 1))
    #     test_data = []
    #     for song in chord_roll:
    #         pos = 0
    #         X = []
    #         while pos+seq_length < song.shape[0]:
    #             sequence = np.array(song[pos:pos+seq_length])
    #             X.append(sequence)
    #             pos += 1
    #
    #         test_data.append(np.array(X))
    #     return np.array(test_data)
    #
    # @staticmethod
    # def NetOutToPianoRoll(network_output, threshold=0.1):
    #     piano_roll = []
    #     for i, timestep in enumerate(network_output):
    #         if np.amax(timestep) > threshold:
    #             pos = 0
    #             pos = np.argmax(timestep)
    #             timestep[:] = np.zeros(timestep.shape)
    #             timestep[pos] = 1
    #         else:
    #             timestep[:] = np.zeros(timestep.shape)
    #         piano_roll.append(timestep)
    #     return np.array(piano_roll)
    #
    # @staticmethod
    # def createMidiFromPianoRoll(piano_roll, lowest_note, directory, mel_test_file, threshold, res_factor=1):
    #
    #     ticks_per_beat = int(96 / res_factor)
    #     mid = MidiFile(type=0, ticks_per_beat=ticks_per_beat)
    #     track = MidiTrack()
    #     mid.tracks.append(track)
    #
    #     mid_files = []
    #
    #     delta_times = [0]
    #     for k in range(piano_roll.shape[1]):  # initial starting values
    #         if piano_roll[0, k] == 1:
    #             track.append(Message('note_on', note=k + lowest_note, velocity=100, time=0))
    #             delta_times.append(0)
    #
    #     for j in range(piano_roll.shape[0] - 1):  # all values between first and last one
    #         set_note = 0  # Check, if for the current timestep a note has already been changed (set to note_on or note_off)
    #
    #         for k in range(piano_roll.shape[1]):
    #             if (piano_roll[j + 1, k] == 1 and piano_roll[j, k] == 0) or (piano_roll[j + 1, k] == 0 and piano_roll[
    #                 j, k] == 1):  # only do something if note_on or note_off are to be set
    #                 if set_note == 0:
    #                     time = j + 1 - sum(delta_times)
    #                     delta_times.append(time)
    #                 else:
    #                     time = 0
    #
    #                 if piano_roll[j + 1, k] == 1 and piano_roll[j, k] == 0:
    #                     set_note += 1
    #                     track.append(Message('note_on', note=k + lowest_note, velocity=100, time=time))
    #                 if piano_roll[j + 1, k] == 0 and piano_roll[j, k] == 1:
    #                     set_note += 1
    #                     track.append(Message('note_off', note=k + lowest_note, velocity=64, time=time))
    #
    #     mid.save('%s%s_th%s.mid' % (directory, mel_test_file, threshold))
    #     mid_files.append('%s.mid' % (mel_test_file))
    #
    #     return
    #
    # # TO BE REMOVED
    #
    # def generate_piano_roll(self):
    #     piano_roll = np.zeros((len(self.files), self.ticks, self.max_note-self.min_note+1), dtype=np.float32)
    #     for i, file_dir in enumerate(self.files):
    #         file_path = "%s" %(file_dir)
    #         mid = MidiFile(file_path)
    #         note_time_onoff = PianoRoll.getNoteTimeOnOffArray(mid, self.res_factor)
    #         note_on_length = PianoRoll.getNoteOnLengthArray(note_time_onoff)
    #         for message in note_on_length:
    #             piano_roll[i, message[1]:(message[1]+int(message[2]/2)), message[0]-self.min_note] = 1
    #     return piano_roll
    #
    # @staticmethod
    # def getNoteTimeOnOffArray(mid, res_factor):
    #     note_time_onoff_array = []
    #     for track in mid.tracks:
    #         current_time = 0
    #         for message in track:
    #             if not isinstance(message, MetaMessage):
    #                 current_time += int(message.time/res_factor)
    #                 if message.type == 'note_on':
    #                     note_onoff = 1
    #                 elif message.type == 'note_off':
    #                     note_onoff = 0
    #                 else:
    #                     print("Error: Note Type not recognized!")
    #                 note_time_onoff_array.append([message.note, current_time, note_onoff])
    #     return note_time_onoff_array
    #
    # @staticmethod
    # def getNoteOnLengthArray(note_time_onoff_array):
    #     note_on_length_array = []
    #     for i, message in enumerate(note_time_onoff_array):
    #         if message[2] == 1: #if note type is 'note_on'
    #             start_time = message[1]
    #             for event in note_time_onoff_array[i:]: #go through array and look for, when the current note is getting turned off
    #                 if event[0] == message[0] and event[2] == 0:
    #                     length = event[1] - start_time
    #                     break
    #
    #             note_on_length_array.append([message[0], start_time, length])
    #     return note_on_length_array
    #
    # @staticmethod
    # def doubleRoll(roll):
    #     double_roll = []
    #     for song in roll:
    #         double_song = np.zeros((roll.shape[1]*2, roll.shape[2]))
    #         double_song[0:roll.shape[1], :] = song
    #         double_song[roll.shape[1]:, :] = song
    #         double_roll.append(double_song)
    #     return np.array(double_roll)
    #
    @staticmethod
    def createNetInputs(roll, target, seq_length=3072):
        #roll: 3-dim array with Midi Files as piano roll. Size: (num_samples=num Midi Files, num_timesteps, num_notes)
        #seq_length: Sequence Length. Length of previous played notes in regard of the current note that is being trained on
        #seq_length in Midi Ticks. Default is 96 ticks per beat --> 3072 ticks = 8 Bars
        X = []
        y = []
        for i, song in enumerate(roll):
            pos = 0
            while pos+seq_length < song.shape[0]:
                sequence = np.array(song[pos:pos+seq_length])
                X.append(sequence)
                y.append(target[i, pos+seq_length])
                pos += 1
        return np.array(X), np.array(y)

    @staticmethod
    def create_test_inputs(roll, seq_length=3072):
        # roll: 3-dim array with Midi Files as piano roll. Size: (num_samples=num Midi Files, num_timesteps, num_notes)
        # seq_length: Sequence Length. Length of previous played notes in regard of the current note that is being trained on
        # seq_length in Midi Ticks. Default is 96 ticks per beat --> 3072 ticks = 8 Bars

        testData = []

        for song in roll:
            pos = 0
            X = []
            while pos + seq_length < song.shape[0]:
                sequence = np.array(song[pos:pos + seq_length])
                X.append(sequence)
                pos += 1

            testData.append(np.array(X))

        return np.array(testData)


        # @staticmethod
    # def fromMidiCreatePianoRoll(files_dir, ticks, lowest_note, column_dim = 12,res_factor=1):
    #     num_files = len(files_dir)
    #
    #     piano_roll = np.zeros((num_files, ticks, column_dim))
    #
    #     for i, file_dir in enumerate(files_dir):
    #         file_path = "%s" %(file_dir)
    #         mid = MidiFile(file_path)
    #         note_time_onoff = PianoRoll.compose_getNoteTimeOnOffArray(mid, res_factor)
    #         note_on_length = PianoRoll.compose_getNoteOnLengthArray(note_time_onoff)
    #         for message in note_on_length:
    #             piano_roll[i, message[1]:(message[1] + message[2]), (message[0]-lowest_note) + 1] = 1
    #
    #     return piano_roll
    #
    # @staticmethod
    # def compose_getNoteTimeOnOffArray(mid, res_factor):
    #
    #     note_time_onoff_array = []
    #
    #     for track in mid.tracks:
    #         current_time = 0
    #         for message in track:
    #             if not isinstance(message, MetaMessage):
    #                 current_time += int(message.time / res_factor)
    #                 if message.type == 'note_on':
    #                     note_onoff = 1
    #                 elif message.type == 'note_off':
    #                     note_onoff = 0
    #                 else:
    #                     print("Error: Note Type not recognized!")
    #
    #                 note_time_onoff_array.append([message.note, current_time, note_onoff])
    #
    #     return note_time_onoff_array
    #
    # @staticmethod
    # def compose_getNoteOnLengthArray(note_time_onoff_array):
    #     note_on_length_array = []
    #     for i, message in enumerate(note_time_onoff_array):
    #         if message[2] == 1:  # if note type is 'note_on'
    #             start_time = message[1]
    #             for event in note_time_onoff_array[i:]:  # go through array and look for, when the current note is getting turned off
    #                 if event[0] == message[0] and event[2] == 0:
    #                     length = event[1] - start_time
    #                     break
    #
    #             note_on_length_array.append([message[0], start_time, length])
    #
    #     return note_on_length_array
