import numpy as np
import torch
from scipy.signal import savgol_filter
from data.dataset_utils import scalar_dB_to_amplitude

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


def mix_song_smooth(dataset, model, loaded_tracks: dict, chunk_length=1, sr=44100) -> np.array:
    chunk_samples = chunk_length * sr
    num_chunks = int(len(loaded_tracks['drums'][0]) / chunk_samples)

    raw_gains = {track: [] for track in ['bass', 'drums', 'vocals', 'other']}

    for chunk_i in range(1, num_chunks):
        i_from = (chunk_i - 1) * chunk_samples
        i_to = chunk_i * chunk_samples

        features = []
        for track in dataset.get_tracklist():
            if track != 'mix':
                feature = dataset.compute_features(loaded_tracks[track][:, i_from:i_to])
                features.append(feature)

        feature_stack = np.stack(features)
        feature_tensor = torch.from_numpy(feature_stack).unsqueeze(0)
        _, gains = model(feature_tensor.to(device))

        for i, track in enumerate(dataset.get_tracklist()):
            if track != 'mix':
                # extra batch dimension -> squeeze
                gain = np.squeeze(gains[i].to('cpu').detach().numpy())
                gain = scalar_dB_to_amplitude(gain)
                raw_gains[track].append(float(gain))

    # any track can be used as a reference, they all have the same length
    # mixed_song = np.zeros_like(loaded_tracks['drums'])
    smooth_gains = {track: [] for track in ['bass', 'drums', 'vocals', 'other']}
    mixed_tracks = {}

    # TODO: if works, remove interpolation and rewrite the pipeline in a separate abstraction
    for track in raw_gains:
        smoothed_gains = savgol_filter(raw_gains[track], 51, 2)
        smooth_gains[track].extend(smoothed_gains)
        mask = interpolate_mask(smoothed_gains, len(loaded_tracks[track][0]))
        # mixed_song += loaded_tracks[track] * mask
        mixed_tracks[track] = loaded_tracks[track] * mask

    assert len(raw_gains['bass']) == len(smooth_gains['bass'])

    mixed_song = np.array(list(mixed_tracks.values()))
    mixed_tracks['mix'] = np.sum(mixed_song, axis=0)

    # mixed_song = librosa.util.normalize(mixed_song, axis=1)

    return mixed_tracks, raw_gains, smooth_gains
