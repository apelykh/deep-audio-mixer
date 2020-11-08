import os
import librosa
import numpy as np


def split_songlist(songlist, train_val_test_split: tuple = (0.8, 0.2, 0.0),
                   summary: bool = True) -> tuple:
    assert sum(train_val_test_split) == 1, 'train/val/test split should sum to 1'

    train_len = round(len(songlist) * train_val_test_split[0])
    val_len = round(len(songlist) * train_val_test_split[1])
    test_len = round(len(songlist) * train_val_test_split[2])

    songlist = set(songlist)

    train_songlist = list(np.random.choice(list(songlist), train_len, replace=False))
    songlist = songlist.difference(set(train_songlist))

    val_songlist = list(np.random.choice(list(songlist), val_len, replace=False))
    songlist = songlist.difference(set(val_songlist))

    test_songlist = list(songlist)

    if summary:
        print('Dataset split:')
        print('=' * 80)
        print('Train: {} tracks'.format(train_len))
        print(train_songlist)
        print('-' * 80)
        print('Val: {} tracks'.format(val_len))
        print(val_songlist)
        print('-' * 80)
        print('Test: {} tracks'.format(test_len))
        print(test_songlist)

    return train_songlist, val_songlist, test_songlist


def scalar_amplitude_to_dB(x):
    """
    amplitude_to_dB(S) = 20 * log10(S)
    """
    return 20 * np.log10(x)


def scalar_dB_to_amplitude(x):
    """
    db_to_amplitude(S_db) ~= 10.0**(0.5 * S_db)
    """
    return np.power(10.0, 0.5 * x)


def load_tracks(base_dir, song_name,
                tracklist=('bass', 'drums', 'vocals', 'other', 'mix'),
                sr=44100) -> dict:
    loaded_tracks = {}

    for track in tracklist:
        if track == 'mix':
            track_path = os.path.join(base_dir, song_name, '{}_MIX.wav'.format(song_name))
        else:
            track_path = os.path.join(base_dir, song_name, '{}_STEMS_JOINED'.format(song_name),
                                      '{}_STEM_{}.wav'.format(song_name, track.upper()))
        # TODO: switch to soundfile.read()
        audio, _ = librosa.load(track_path, sr=sr, mono=False)
        loaded_tracks[track] = audio

    return loaded_tracks


def load_tracks_musdb18(base_dir, song_name,
                        tracklist=('bass', 'drums', 'vocals', 'other', 'mix'),
                        sr=44100) -> dict:
    loaded_tracks = {}

    for track in tracklist:
        track_name = 'mixture' if track == 'mix' else track
        track_path = os.path.join(base_dir, song_name, '{}.wav'.format(track_name))
        # TODO: switch to soundfile.read()
        audio, _ = librosa.load(track_path, sr=sr, mono=False)
        loaded_tracks[track] = audio

    return loaded_tracks
