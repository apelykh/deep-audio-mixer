import os
import soundfile as sf
import librosa
import torch
import json
import numpy as np
import pyloudnorm as pyln
from statistics import mean
from collections import OrderedDict
from inference_utils import mix_song_smooth
from data.dataset import MultitrackAudioDataset
from models.model_scalar_v2 import MixingModelScalar2d
from data.dataset_utils import load_tracks_musdb18


class LoudnessEvaluator:
    def __init__(self, dataset, model, seed=123):
        if seed:
            np.random.seed(seed)

        self.d = dataset
        self.model = model
        self.results_dir = './experiment'
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def process_song(self, loaded_tracks: dict, song_name, n_experiments=5):
        keys = ['bass', 'drums', 'vocals', 'other']
        stats = {
            'song_name': song_name,
            'reference': {},
            'sums': [],
            'mixes': []
        }

        track_sum = np.array(list(loaded_tracks.values()))
        track_sum = np.sum(track_sum, axis=0)
        track_sum = librosa.util.normalize(track_sum, axis=1)
        sf.write(os.path.join(self.results_dir, '{}_reference.wav'.format(song_name)), track_sum.T, 44100)

        ref_loudness = LoudnessEvaluator.evaluate_loudness(loaded_tracks)
        loudness_dict = OrderedDict(zip(keys, ref_loudness))
        stats['reference'] = loudness_dict

        for exp_i in range(n_experiments):
            print('Run {}/{}'.format(exp_i, n_experiments))
            random_gains = np.random.uniform(0.3, 1.7, size=4)

            gained_tracks = {}
            for i, track in enumerate(loaded_tracks):
                if track != 'mix':
                    gained_tracks[track] = random_gains[i] * loaded_tracks[track]

            gained_sum = np.array(list(gained_tracks.values()))
            gained_sum = np.sum(gained_sum, axis=0)
            gained_tracks['mix'] = gained_sum

            gained_sum = librosa.util.normalize(gained_sum, axis=1)
            sf.write(os.path.join(self.results_dir, '{}_sum_{}.wav'.format(song_name, exp_i)), gained_sum.T, 44100)

            # 1. loudness of gained tracks: supposed to be further from the reference
            gained_loudness = LoudnessEvaluator.evaluate_loudness(gained_tracks)
            loudness_dict = OrderedDict(zip(keys, gained_loudness))
            stats['sums'].append(loudness_dict)

            # 2. loudness of mixed tracks
            mixed_tracks, _, _ = mix_song_smooth(self.d, self.model, gained_tracks)
            mix_loudness = LoudnessEvaluator.evaluate_loudness(mixed_tracks)
            loudness_dict = OrderedDict(zip(keys, mix_loudness))
            stats['mixes'].append(loudness_dict)

            mix = librosa.util.normalize(mixed_tracks['mix'], axis=1)
            sf.write(os.path.join(self.results_dir, '{}_mixed_{}.wav'.format(song_name, exp_i)), mix.T, 44100)

        return stats

    @staticmethod
    def _calculate_diff_between_loudness_dicts(l_dict1: OrderedDict, l_dict2: OrderedDict):
        a1 = np.array(list(l_dict1.values()))
        a2 = np.array(list(l_dict2.values()))
        diffs = np.abs(a1 - a2)
        # sum of absolute differences
        return float(np.sum(diffs))

    def postprocess_song_stats(self, stats: dict):
        mix_errors = []
        sum_errors = []

        for run_i in range(len(stats['mixes'])):
            sum_ref_error = LoudnessEvaluator.\
                _calculate_diff_between_loudness_dicts(stats['sums'][run_i], stats['reference'])
            mix_ref_error = LoudnessEvaluator. \
                _calculate_diff_between_loudness_dicts(stats['mixes'][run_i], stats['reference'])

            mix_errors.append(mix_ref_error)
            sum_errors.append(sum_ref_error)
            print('Run {}: sum error: {:.5f}, mix_error: {:.5f}'.
                  format(run_i, sum_ref_error, mix_ref_error))

        print('-' * 80)
        print('Mean sum error: {:.5f}, mean mix error: {:.5f}'.
              format(mean(sum_errors), mean(mix_errors)))

    @staticmethod
    def evaluate_loudness(tracks: dict) -> list:
        per_track_loudness = []
        meter = pyln.Meter(44100, filter_class="Fenton/Lee 1")  # create BS.1770 meter

        mix_loudness = meter.integrated_loudness(tracks['mix'].T)

        for track_name in tracks:
            if track_name != 'mix':
                track_loudness = meter.integrated_loudness(tracks[track_name].T)
                # print('{} loudness: {:.5f}'.format(track_name.upper(), track_loudness))
                per_track_loudness.append(track_loudness / mix_loudness)

        return per_track_loudness


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Torch version: ', torch.__version__)
    print('Device: ', device)

    base_path = '/media/apelykh/bottomless-pit/datasets/mixing/MedleyDB/Audio'
    seed = 321
    chunk_length = 1

    d = MultitrackAudioDataset(
        base_path,
        songlist=[],
        chunk_length=chunk_length,
        normalize=False,
        compute_features=False
    )

    model = MixingModelScalar2d().to(device)
    weights = './saved_models/training-ignite-unnorm-70-epochs-135.08-val-loss-fter-bugfix/scalar2d_scalar2d_5658.pt'
    model.load_state_dict(torch.load(weights, map_location=device))

    base_dir = '/media/apelykh/bottomless-pit/datasets/mixing/MUSDB18HQ/test/'
    song_name = 'Moosmusic - Big Dummy Shake'
    loaded_tracks = load_tracks_musdb18(base_dir, song_name)

    l_eval = LoudnessEvaluator(d, model)
    stats = l_eval.process_song(OrderedDict(loaded_tracks), song_name, n_experiments=3)

    print(json.dumps(stats, indent=4))

    l_eval.postprocess_song_stats(stats)
