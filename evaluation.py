import os
import soundfile as sf
import librosa
import torch
import json
import pickle
import numpy as np
import pyloudnorm as pyln
from statistics import mean
from collections import OrderedDict
from inference_utils import mix_song_smooth
from data.dataset import MultitrackAudioDataset
from data.dataset_utils import load_tracks_musdb18
from data.medleydb_split import weathervane_music

from models.model_scalar_1s import MixingModelScalar1s
from models.model_scalar_2s import MixingModelScalar2s
from models.baselines.mean_loudness_model import MeanLoudnessModel
from models.baselines.random_model import RandomModel


class LoudnessEvaluator:
    def __init__(self, dataset, d_mean_loudness, mix_model, sr=44100, seed=None):
        if seed:
            np.random.seed(seed)

        self.sr = sr
        self.d = dataset
        self.mix_model = mix_model
        self.mean_loudness_model = MeanLoudnessModel(d_mean_loudness)
        self.random_model = RandomModel()

        self.keys = ('bass', 'drums', 'vocals', 'other')

        self.results_dir = './experiment'
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def evaluate_loudness(self, tracks: dict) -> list:
        assert 'mix' in tracks, 'Mix has to be in track dict for evaluation'

        per_track_loudness = []
        meter = pyln.Meter(self.sr)

        mix_loudness = meter.integrated_loudness(tracks['mix'].T)

        for track_name in self.keys:
            track_loudness = meter.integrated_loudness(tracks[track_name].T)
            # print('{} loudness: {:.5f}'.format(track_name.upper(), track_loudness))
            per_track_loudness.append(track_loudness / mix_loudness)

        return per_track_loudness

    def _sum_and_evaluate_tracks(self, track_dict, song_name, identifier, add_mix_to_dict=True,
                                 write_to_disc=True) -> dict:
        track_arr = np.array(list(track_dict.values()))
        track_sum = np.sum(track_arr, axis=0)

        if add_mix_to_dict:
            track_dict['mix'] = track_sum

        if write_to_disc:
            track_sum = librosa.util.normalize(track_sum, axis=1)
            sf.write(os.path.join(self.results_dir, '{}_{}.wav'.format(song_name, identifier)),
                     track_sum.T, self.sr)

        ref_loudness = self.evaluate_loudness(track_dict)
        loudness_dict = OrderedDict(zip(self.keys, ref_loudness))

        return loudness_dict

    def process_song(self, loaded_tracks: dict, song_name: str) -> dict:
        stats = dict()

        # naive sum of loaded multitracks - best mix (MUSDB18)
        stats['reference'] = self._sum_and_evaluate_tracks(loaded_tracks, song_name, 'reference', False)

        # sum of multitracks with random gains
        random_mix_tracks = self.random_model.forward(loaded_tracks)
        stats['random'] = self._sum_and_evaluate_tracks(random_mix_tracks, song_name, 'random')

        # each multitrack is normalized to the mean loudness of the corresponding track from train set
        loudnorm_tracks = self.mean_loudness_model.forward(loaded_tracks)
        stats['loudnorm'] = self._sum_and_evaluate_tracks(loudnorm_tracks, song_name, 'loudnorm')

        # mixed by the model
        mixed_tracks, _, _ = mix_song_smooth(self.d, self.mix_model, random_mix_tracks, chunk_length=1)
        stats['mix'] = self._sum_and_evaluate_tracks(mixed_tracks, song_name, 'mix')

        return stats

    @staticmethod
    def _calculate_diff_between_loudness_dicts(l_dict1: OrderedDict, l_dict2: OrderedDict):
        a1 = np.array(list(l_dict1.values()))
        a2 = np.array(list(l_dict2.values()))
        diffs = np.abs(a1 - a2)
        # sum of absolute differences
        return float(np.sum(diffs))

    def postprocess_song_stats(self, stats: dict):
        random_error = LoudnessEvaluator.\
            _calculate_diff_between_loudness_dicts(stats['random'], stats['reference'])
        loudnorm_error = LoudnessEvaluator. \
            _calculate_diff_between_loudness_dicts(stats['loudnorm'], stats['reference'])
        mix_error = LoudnessEvaluator. \
            _calculate_diff_between_loudness_dicts(stats['mix'], stats['reference'])

        print('Random error: {:.5f}\nloudnorm error: {:.5f}\nmix_error: {:.5f}'.
              format(random_error, loudnorm_error, mix_error))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Torch version: ', torch.__version__)
    print('Device: ', device)

    base_path = '/media/apelykh/bottomless-pit/datasets/mixing/MedleyDB/Audio'
    seed = 321
    chunk_length = 1

    train_songlist = ['PortStWillow_StayEven', 'HeladoNegro_MitadDelMundo', 'Lushlife_ToynbeeSuite', 'FamilyBand_Again',
                      'SweetLights_YouLetMeDown', 'AvaLuna_Waterduct', 'TheSoSoGlos_Emergency', 'PurlingHiss_Lolita',
                      'TheDistricts_Vermont', 'BigTroubles_Phantom', 'AClassicEducation_NightOwl',
                      'Auctioneer_OurFutureFaces', 'InvisibleFamiliars_DisturbingWildlife', 'SecretMountains_HighHorse',
                      'Grants_PunchDrunk', 'Snowmine_Curfews', 'NightPanther_Fire', 'HezekiahJones_BorrowedHeart',
                      'DreamersOfTheGhetto_HeavyLove', 'HopAlong_SisterCities']

    val_songlist = ['StevenClark_Bounty', 'StrandOfOaks_Spacestation', 'CelestialShore_DieForUs',
                    'FacesOnFilm_WaitingForGa', 'Creepoid_OldTree']

    d = MultitrackAudioDataset(
        base_path,
        songlist=train_songlist,
        chunk_length=chunk_length,
        normalize=False,
        compute_features=True
    )

    if not os.path.exists('./mean_loudness.pkl'):
        mean_loudness = d.compute_mean_loudness()
        with open("mean_loudness.pkl", "wb") as f:
            pickle.dump(mean_loudness, f)
    else:
        with open("mean_loudness.pkl", "rb") as f:
            mean_loudness = pickle.load(f)

    model = MixingModelScalar1s().to(device)
    weights = './saved_models/from_server/20-08-2020-20:06_training_4masks_unnorm_1s_medleydb/best_scalar1s_62_neg_train_loss=-130.4279.pt'
    model.load_state_dict(torch.load(weights, map_location=device))

    base_dir = '/media/apelykh/bottomless-pit/datasets/mixing/MUSDB18HQ/test/'
    song_name = 'PR - Oh No'
    loaded_tracks = load_tracks_musdb18(base_dir, song_name)

    l_eval = LoudnessEvaluator(d, mean_loudness, model)
    stats = l_eval.process_song(OrderedDict(loaded_tracks), song_name)

    print(json.dumps(stats, indent=4))
    l_eval.postprocess_song_stats(stats)
