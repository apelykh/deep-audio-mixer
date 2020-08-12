import os
import math
import numpy as np
import librosa
import librosa.display
import torch

from scipy.signal import savgol_filter
from data.dataset import MultitrackAudioDataset
from models.model_dummy import ModelDummy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def interpolate_mask(spec_mask: np.array, tgt_len: int) -> np.array:
    """
    Interpolate mask to a target length.
    Used for stretching shorter spectral-level masks to sample-level size.

    :param spec_mask: a mask that will be interpolated;
    :param tgt_len: target length of the mask;
    :return: stretched mask of the target length;
    """
    assert len(spec_mask) <= tgt_len, "Target mask should be longer than the initial one"

    sample_mask = np.zeros(tgt_len)

    # num of samples influenced by each value in the initial mask
    interp_coef = int(tgt_len / len(spec_mask))

    # after the loop finishes, will hold a start index of the last chunk
    final_i = -1
    for chunk_i in range(0, len(spec_mask) - 1):
        i_from = chunk_i * interp_coef
        i_to = (chunk_i + 1) * interp_coef
        sample_mask[i_from:i_to] = spec_mask[chunk_i]
        final_i = i_to

    # last value from spec_mask is filling the last chunk
    # and goes beyond, to the end of sample_mask
    if final_i > -1:
        sample_mask[final_i:] = spec_mask[-1]

    return sample_mask


def mix_song(dataset, model, loaded_tracks: dict, chunk_length=1, sr=44100) -> np.array:
    """
    Sequentially apply the model to all the song chunks to produce the full mixed song.

    :param dataset: an instance of a MultitrackAudioDataset class;
    :param model: an instance of MixingModelScalar or MixingModelVector classes;
    :param loaded_tracks: a dict with audio tracks:
    {
        'bass': np.array,
        'drums': np.array,
        'other': np.array,
        'vocals': np.array,
        'mix': np.array - optional
    }
    :param chunk_length: length (in seconds) of the audio chunk to compute features for;
        Should correspond to the value used for training;
    :param sr: sampling rate of the audio tracks;
    :return: full mixed song;
    """
    # any track can be used as a reference, they all have the same length
    mixed_song = np.zeros(len(loaded_tracks['drums']))
    chunk_samples = chunk_length * sr
    num_chunks = int(len(loaded_tracks['drums']) / chunk_samples)

    mask_history = {track: [] for track in ['bass', 'drums', 'vocals', 'other']}

    for chunk_i in range(1, num_chunks):
        i_from = (chunk_i - 1) * chunk_samples
        i_to = chunk_i * chunk_samples

        features = []
        for track in dataset.get_tracklist():
            if track != 'mix':
                feature, _ = dataset.compute_features(loaded_tracks[track][i_from:i_to])
                features.append(feature)

        # stack spectrograms of all tracks the same way we did during training
        feature_stack = np.stack(features)
        # adding a "batch" dimension
        feature_tensor = torch.from_numpy(feature_stack).unsqueeze(0)
        # feature_tensor = torch.from_numpy(feature_stack)
        # obtain gain masks for the current chunk
        _, masks = model(feature_tensor.to(device))

        mixed_chunk = np.zeros(chunk_samples)
        for i, track in enumerate(dataset.get_tracklist()):
            if track != 'mix':
                # extra batch dimension -> squeeze
                spec_mask = np.squeeze(masks[i].to('cpu').detach().numpy())
                mask_history[track].append(float(spec_mask))
                # a hacky way to differentiate between 1d mask and a scalar value
                if spec_mask.shape:
                    sample_mask = interpolate_mask(spec_mask, chunk_samples)
                else:
                    sample_mask = spec_mask
                mixed_chunk += loaded_tracks[track][i_from:i_to] * sample_mask
        mixed_song[i_from:i_to] = mixed_chunk

    return mixed_song, mask_history


def _dB_to_amplitude(x):
    """
    db_to_amplitude(S_db) ~= 10.0**(0.5 * S_db)
    """
    return np.power(10.0, 0.5 * x)


def mix_song_smooth(dataset, model, loaded_tracks: dict, chunk_length=1, sr=44100) -> np.array:
    # any track can be used as a reference, they all have the same length
    mixed_song = np.zeros(len(loaded_tracks['drums']))
    chunk_samples = chunk_length * sr
    num_chunks = int(len(loaded_tracks['drums']) / chunk_samples)

    gain_history = {track: [] for track in ['bass', 'drums', 'vocals', 'other']}

    for chunk_i in range(1, num_chunks):
        i_from = (chunk_i - 1) * chunk_samples
        i_to = chunk_i * chunk_samples

        features = []
        for track in dataset.get_tracklist():
            if track != 'mix':
                feature = dataset.compute_features(loaded_tracks[track][i_from:i_to])
                features.append(feature)

        feature_stack = np.stack(features)
        feature_tensor = torch.from_numpy(feature_stack).unsqueeze(0)
        _, gains = model(feature_tensor.to(device))

        for i, track in enumerate(dataset.get_tracklist()):
            if track != 'mix':
                # extra batch dimension -> squeeze
                gain = np.abs(np.squeeze(gains[i].to('cpu').detach().numpy()))
                gain = _dB_to_amplitude(gain)
                gain_history[track].append(float(gain))

    # TODO: if works, remove interpolation and rewrite the pipeline in a separate abstraction
    for track in gain_history:
        smoothed_gains = savgol_filter(gain_history[track], 51, 2)
        mask = interpolate_mask(smoothed_gains, len(loaded_tracks[track]))
        mixed_song += loaded_tracks[track] * mask

    mixed_song = librosa.util.normalize(mixed_song)

    return mixed_song, gain_history


def mix_song_istft(dataset, model, loaded_tracks: dict, chunk_length=1, sr=44100) -> np.array:
    # for now inference is limited to a single song at a time
    assert dataset.get_num_songs() == 1

    mixed_song = np.zeros(len(loaded_tracks['drums']))
    chunk_samples = chunk_length * sr
    num_chunks = int(len(loaded_tracks['drums']) / chunk_samples)

    for chunk_i in range(1, num_chunks):
        i_from = (chunk_i - 1) * chunk_samples
        i_to = chunk_i * chunk_samples

        features = []
        sum_chunk = np.zeros(chunk_samples)
        for track in dataset.get_tracklist():
            if track != 'mix':
                track_chunk = loaded_tracks[track][i_from:i_to]
                magnitudes, _ = dataset.compute_features(track_chunk)
                features.append(magnitudes)
                sum_chunk += track_chunk

        # obtain phases of a summed track to use then in istft
        _, sum_chunk_phases = dataset.compute_features(sum_chunk)

        # stack spectrograms of all tracks the same way we did during training
        feature_stack = np.stack(features)
        # adding a "batch" dimension
        feature_tensor = torch.from_numpy(feature_stack).unsqueeze(0)
        masked = model(feature_tensor.to(device))
        masked = masked.to('cpu').detach().numpy()
        masked = librosa.db_to_amplitude(masked[0])

        # TODO: parametrize istft from outside the func
        mixed_chunk = librosa.core.istft(masked * sum_chunk_phases,
                                         hop_length=512,
                                         win_length=2048,
                                         length=chunk_samples)
        mixed_song[i_from:i_to] = mixed_chunk

    return mixed_song


def mix_dataset_istft(dataset, model) -> np.array:
    # for now inference is limited to a single song at a time
    assert dataset.get_num_songs() == 1

    mixed_song = []
    for chunk_i in range(len(dataset)):
        chunk = dataset[chunk_i]
        feature_tensor = torch.from_numpy(chunk['train_features']).unsqueeze(0)

        masked = model(feature_tensor.to(device))
        masked = masked.to('cpu').detach().numpy()
        masked = librosa.db_to_amplitude(masked[0])

        mixed_chunk = librosa.core.istft(masked * chunk['sum_phases'],
                                         hop_length=512,
                                         win_length=2048)
        mixed_song.append(mixed_chunk)

    return np.concatenate(mixed_song, axis=None)


def load_tracks(base_dir, song_name, tracklist=('bass', 'drums', 'vocals', 'other', 'mix'),
                sr=44100) -> dict:
    loaded_tracks = {}

    for track in tracklist:
        if track == 'mix':
            track_path = os.path.join(base_dir, song_name, '{}_MIX.wav'.format(song_name))
        else:
            track_path = os.path.join(base_dir, song_name, '{}_STEMS_JOINED'.format(song_name),
                                      '{}_STEM_{}.wav'.format(song_name, track.upper()))
        audio, _ = librosa.load(track_path, sr=sr)
        # loaded_tracks[track] = librosa.util.normalize(audio)
        loaded_tracks[track] = audio

    return loaded_tracks


def load_tracks_musdb18(base_dir, song_name, tracklist=('bass', 'drums', 'vocals', 'other', 'mix'),
                sr=44100) -> dict:
    loaded_tracks = {}

    for track in tracklist:
        track_name = 'mixture' if track == 'mix' else track
        track_path = os.path.join(base_dir, song_name, '{}.wav'.format(track_name))
        audio, _ = librosa.load(track_path, sr=sr)
        loaded_tracks[track] = librosa.util.normalize(audio)

    return loaded_tracks


if __name__ == '__main__':
    d = MultitrackAudioDataset(
        '/media/apelykh/bottomless-pit/datasets/mixing/MedleyDB/Audio',
        # '/home/apelykh/datasets',
        songlist=['Creepoid_OldTree'],
        # songlist=['PurlingHiss_Lolita'],
        chunk_length=1,
        train_val_test_split=(0.0, 0.0, 1.0),
        mode='test',
        seed=321,
        normalize=False
    )
    # print('Test: {} tracks, {} chunks'.format(d.get_num_songs(), len(d)))

    base_dir = '/media/apelykh/bottomless-pit/datasets/mixing/MedleyDB/Audio/'
    song_name = 'TheScarletBrand_LesFleursDuMal'
    loaded_tracks = load_tracks(base_dir, song_name)

    # model = MixingModelTDD().to(device)
    model = ModelDummy().to(device)
    # model = ModelUNet(n_channels=4, n_classes=4, bilinear=False).to(device)

    num_trainable_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('{} trainable parameters'.format(num_trainable_param))

    mixed_song = mix_song_istft(d, model, loaded_tracks)

    librosa.output.write_wav('results/test.wav', mixed_song, 44100)
