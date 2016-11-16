"""
Author: Arulselvan Madhavan
github: https://github.com/Arulselvanmadhavan
"""
import numpy as np
from mido import MidiFile, MidiTrack, Message
from mido import MetaMessage
from Utils import Utils
from functools import partial
import warnings

# GLOBAL CONSTANTS
MIDI_FILE_EXTENSION = "mid"
NOTE_ON = "note_on"
NOTE_OFF = "note_off"
RES_FACTOR = "res_factor"
MAX_NOTE = "max_note"
MIN_NOTE = "min_note"
TICKS = "ticks"
MIN_NOTE_INITIALIZER = 10000
MAX_NOTE_INITIALIZER = 0
MAX_TICK_INITIALIZER = 0

class PianoRoll(object):
    """Piano Roll Representation."""

    def __init__(self, path_to_data, res_factor=12):
        Utils.validate_path(path_to_data)
        self.path = path_to_data
        self.res_factor = res_factor
        self.files = Utils.get_matching_files(self.path+"*."+MIDI_FILE_EXTENSION)
        self.ticks, self.max_note, self.min_note = self.collect_stats(self.files, self.res_factor)

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
                       (MAX_TICK_INITIALIZER, MAX_NOTE_INITIALIZER, MIN_NOTE_INITIALIZER))
        return stats[0], stats[1], stats[2]

    @staticmethod
    def get_note_info(file_path, configs):
        mid = MidiFile(file_path)
        track_notes = map(partial(PianoRoll.get_note_on_off_info, configs=configs),
                          mid.tracks)
        if len(track_notes) > 1 : #TODO - DEBUG Block Remove later.
            print("More than 1 track in {}".format(file_path))
        return np.vstack(track_notes)

    @staticmethod
    def get_note_on_off_info(track, configs):
        # notes = np.zeros((len(track), 3))
        # result = reduce(partial(PianoRoll.parse_midi_message, configs=configs),
        #                 track,
        #                 (0, 0, notes))
        # return np.array(notes[:result[0], :]) #Filter non-zero rows(count given by result[0])

        result = reduce(partial(PianoRoll.get_note_length, configs=configs),
                track,
                (0, [], [-1] * (configs[MAX_NOTE] - configs[MIN_NOTE] + 1)))
        return result[1]

    @staticmethod
    def parse_midi_message(result, message, configs):
        index, current_time, notes = result
        if not isinstance(message, MetaMessage):
            current_time += int(message.time/configs[RES_FACTOR])
            if message.type == NOTE_ON:
                note_onoff = 1
            elif message.type == NOTE_OFF:
                note_onoff = 0
            else:
                print("Error: Note Type not recognized!")
            notes[index] = [message.note, current_time, note_onoff]
            index += 1
        return (index, current_time, notes)

    @staticmethod
    def get_note_length(result, msg, configs):
        """
        :param result: (current_time, notes, start_time_tracker)
        :param msg: MidiMessage
        :param configs:
        :return:
        """
        current_time, notes, start_time_tracker = result
        if not isinstance(msg, MetaMessage):
            resolved_time = int(msg.time/configs[RES_FACTOR])
            current_time += resolved_time
            if msg.type == NOTE_ON:
                start_time_tracker[msg.note - configs[MIN_NOTE]] = current_time
            elif msg.type == NOTE_OFF:
                start_time = start_time_tracker[msg.note - configs[MIN_NOTE]]
                if start_time == -1:
                    warnings.warn("End note occurs without a start")
                notes.append([msg.note, start_time, current_time - start_time])
        return (current_time, notes, start_time_tracker)

    def generate_piano_roll_func(self):
        configs = {
            RES_FACTOR : self.res_factor,
            MAX_NOTE : self.max_note,
            MIN_NOTE : self.min_note
        }
        files_notes_on_length = map(partial(PianoRoll.get_note_info, configs=configs),
                                    self.files)
        # file_notes_array = np.array(files_notes_on_length, dtype=np.float32)
        roll_configs = {
            TICKS : self.ticks,
            MAX_NOTE: self.max_note,
            MIN_NOTE: self.min_note
        }
        piano_roll_collection = map(partial(PianoRoll.piano_notes_wrapper, configs=roll_configs),
                                    files_notes_on_length)
        # for i in range(len(piano_roll_collection)):
        #     print(piano_roll_collection[i].shape)
        final = np.vstack(piano_roll_collection)
        print(final.shape)
        return final

    @staticmethod
    def piano_notes_wrapper(notes_per_file, configs):
        piano_roll_per_file = np.zeros((1, configs[TICKS], configs[MAX_NOTE]-configs[MIN_NOTE]+1), dtype=np.float32)
        return reduce(partial(PianoRoll.enter_piano_notes, min_note=configs[MIN_NOTE]),
               notes_per_file,
               piano_roll_per_file)

    @staticmethod
    def enter_piano_notes(piano_roll, msg, min_note):
        """
        Update the piano_roll with the stats from notes
        :param piano_roll:
        :param note_stat:
        :param configs:
        :return:
        """
        piano_roll[0, msg[1]:(msg[1]+int(msg[2]/2)), msg[0]-min_note] = 1
        return piano_roll

    def generate_piano_roll(self):
        piano_roll = np.zeros((len(self.files), self.ticks, self.max_note-self.min_note+1), dtype=np.float32)
        for i, file_dir in enumerate(self.files):
            file_path = "%s" %(file_dir)
            mid = MidiFile(file_path)
            note_time_onoff = PianoRoll.getNoteTimeOnOffArray(mid, self.res_factor)
            note_on_length = PianoRoll.getNoteOnLengthArray(note_time_onoff)
            for message in note_on_length:
                piano_roll[i, message[1]:(message[1]+int(message[2]/2)), message[0]-self.min_note] = 1
        return piano_roll

    @staticmethod
    def getNoteTimeOnOffArray(mid, res_factor):
        note_time_onoff_array = []
        for track in mid.tracks:
            current_time = 0
            for message in track:
                if not isinstance(message, MetaMessage):
                    current_time += int(message.time/res_factor)
                    if message.type == 'note_on':
                        note_onoff = 1
                    elif message.type == 'note_off':
                        note_onoff = 0
                    else:
                        print("Error: Note Type not recognized!")
                    note_time_onoff_array.append([message.note, current_time, note_onoff])
        return note_time_onoff_array

    @staticmethod
    def getNoteOnLengthArray(note_time_onoff_array):
        note_on_length_array = []
        for i, message in enumerate(note_time_onoff_array):
            if message[2] == 1: #if note type is 'note_on'
                start_time = message[1]
                for event in note_time_onoff_array[i:]: #go through array and look for, when the current note is getting turned off
                    if event[0] == message[0] and event[2] == 0:
                        length = event[1] - start_time
                        break

                note_on_length_array.append([message[0], start_time, length])
        return note_on_length_array