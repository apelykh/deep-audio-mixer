import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def interpolate_mask(spec_mask: np.ndarray, tgt_len: int) -> np.ndarray:
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


def mix_song(dataset, model, loaded_tracks: dict, chunk_length=5, sr=44100,
             dummy_masks=False) -> np.ndarray:
    mixed_song = np.zeros(len(loaded_tracks['mixture']))
    chunk_samples = chunk_length * sr
    num_chunks = int(len(loaded_tracks['mixture']) / chunk_samples)

    for chunk_i in range(1, num_chunks):
        i_from = (chunk_i - 1) * chunk_samples
        i_to = chunk_i * chunk_samples

        features = []
        for track in dataset.get_tracklist():
            if track != 'mixture':
                feature = dataset.compute_features(loaded_tracks[track][i_from:i_to])
                features.append(feature)

        feature_stack = np.stack(features)
        feature_tensor = torch.Tensor(feature_stack[np.newaxis, :])
        _, masks = model(feature_tensor.to(device))

        mixed_chunk = np.zeros(chunk_samples)
        for i, track in enumerate(dataset.get_tracklist()):
            if track != 'mixture':
                # extra batch dimension -> squeeze
                spec_mask = np.squeeze(masks[i].to('cpu').detach().numpy())

                # testing the function: sum all the tracks with equal weights
                if dummy_masks:
                    spec_mask = np.ones_like(spec_mask)

                sample_mask = interpolate_mask(spec_mask, chunk_samples)
                mixed_chunk += loaded_tracks[track][i_from:i_to] * sample_mask

        mixed_song[i_from:i_to] = mixed_chunk

    return mixed_song
