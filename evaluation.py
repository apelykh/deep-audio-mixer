import os
import csv
import soundfile as sf
import librosa
import torch
import json
import pickle
import numpy as np
import pyloudnorm as pyln
from openpyxl import Workbook
from statistics import mean
from collections import OrderedDict
from inference_utils import mix_song_smooth
from data.dataset import MultitrackAudioDataset
from data.dataset_utils import load_tracks_musdb18, load_tracks
from data.songlists import medleydb_weathervane_music

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

        self.meter = pyln.Meter(sr)
        self.keys = ('bass', 'drums', 'vocals', 'other')

        self.results_dir = './experiment'
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def evaluate_loudness(self, tracks: dict) -> list:
        per_track_loudness = [self.meter.integrated_loudness(tracks[track_name].T)
                              for track_name in self.keys]
        avg_loudness = mean(per_track_loudness)
        per_track_loudness_norm = [track_loudness - avg_loudness
                                   for track_loudness in per_track_loudness]

        return per_track_loudness_norm

    @staticmethod
    def _calculate_diff_between_loudness_dicts(l_dict1: OrderedDict, l_dict2: OrderedDict):
        a1 = np.array(list(l_dict1.values()))
        a2 = np.array(list(l_dict2.values()))
        diffs = np.abs(a1 - a2)
        # sum of absolute differences
        return float(np.mean(diffs))

    def _sum_and_evaluate_tracks(self, track_dict, reference_dict, song_name, identifier, write_to_disk=True):
        print('[.] {}'.format(identifier.upper()))

        track_arr = np.array(list(track_dict.values()))
        track_sum = np.sum(track_arr, axis=0)

        if write_to_disk:
            track_sum = librosa.util.normalize(track_sum, axis=1)
            sf.write(os.path.join(self.results_dir, '{}_{}.wav'.format(song_name, identifier)),
                     track_sum.T, self.sr)

        per_track_loudness = self.evaluate_loudness(track_dict)
        loudness_dict = OrderedDict(zip(self.keys, per_track_loudness))

        if reference_dict:
            error = LoudnessEvaluator. \
                _calculate_diff_between_loudness_dicts(loudness_dict, reference_dict)
            return loudness_dict, error
        return loudness_dict, None

    def evaluate_medleyb_song(self, loaded_tracks: dict, song_name: str):
        # 1. mix multitracks, check loudness
        # 2. apply mean loudness model, check loudness

        loudnorm_tracks = self.mean_loudness_model.forward(loaded_tracks)
        loudnorm_loudness, _ = self._sum_and_evaluate_tracks(loudnorm_tracks, None, song_name,
                                                             identifier='reference',
                                                             write_to_disk=True)
        print(loudnorm_loudness)
        print('-' * 80)

        mixed_tracks, _, _ = mix_song_smooth(self.d, self.mix_model, loaded_tracks, chunk_length=1)
        mixed_loudness, _ = self._sum_and_evaluate_tracks(mixed_tracks, None, song_name,
                                                          identifier='reference',
                                                          write_to_disk=True)
        print(mixed_loudness)

    def process_song(self, base_dir: str, song_name: str, n_random_samples: int = 5,
                     write_wavs_to_disk=False) -> dict:
        stats = {
            'song_name': song_name
        }

        # loading reference
        loaded_tracks = load_tracks_musdb18(base_dir, song_name + '_gain_mixed')
        reference, _ = self._sum_and_evaluate_tracks(loaded_tracks, None, song_name,
                                                     identifier='reference',
                                                     write_to_disk=write_wavs_to_disk)

        loaded_tracks = load_tracks_musdb18(base_dir, song_name)
        _, stats['sum_error'] = self._sum_and_evaluate_tracks(loaded_tracks, reference, song_name,
                                                              identifier='sum',
                                                              write_to_disk=write_wavs_to_disk)

        # each multitrack is normalized to the mean loudness of the corresponding track from train set
        loudnorm_tracks = self.mean_loudness_model.forward(loaded_tracks)
        _, stats['loudnorm_error'] = self._sum_and_evaluate_tracks(loudnorm_tracks, reference, song_name,
                                                                   identifier='loudnorm',
                                                                   write_to_disk=write_wavs_to_disk)

        # mixed by the model
        mixed_tracks, _, _ = mix_song_smooth(self.d, self.mix_model, loaded_tracks, chunk_length=1)
        _, stats['mix_error'] = self._sum_and_evaluate_tracks(mixed_tracks, reference, song_name,
                                                              identifier='mix',
                                                              write_to_disk=write_wavs_to_disk)
        random_errors = []

        for exp_i in range(n_random_samples):
            # sum of multitracks with random gains
            random_mix_tracks = self.random_model.forward(loaded_tracks)
            _, random_error = self._sum_and_evaluate_tracks(random_mix_tracks, reference, song_name,
                                                            identifier='random_{}'.format(exp_i),
                                                            write_to_disk=write_wavs_to_disk)
            random_errors.append(random_error)
        stats['random_error'] = mean(random_errors)

        # print('Random error: {:.5f}\nloudnorm error: {:.5f}\nmix_error: {:.5f}\nsum_error: {:.5f}'.
        #       format(stats['random_error'], stats['loudnorm_error'], stats['mix_error'], stats['sum_error']))

        return stats

    def _write_xls_row(self, sheet, data: list, row_i: int):
        for i, value in enumerate(data):
            sheet.cell(column=i + 1, row=row_i, value=value)

    def process_songlist(self, base_dir, songlist, write_to_disk=False):
        book = Workbook()
        sheet = book.active

        keys = ['song_name', 'random_error', 'loudnorm_error', 'mix_error']
        self._write_xls_row(sheet, keys, 1)

        random_errors = []
        loudnorm_errors = []
        mix_errors = []

        for i, song_name in enumerate(songlist):
            print('{}/{}: {}'.format(i + 1, len(songlist), song_name))
            loaded_tracks = load_tracks_musdb18(base_dir, song_name)
            stats = self.process_song(OrderedDict(loaded_tracks), song_name,
                                      write_wavs_to_disk=write_to_disk)

            # data = ['{:.4f}'.format(elem) if type(elem) == 'float' else elem for elem in stats.values()]
            # self._write_xls_row(sheet, data, i + 2)

            sheet.cell(column=1, row=i + 2, value=stats['song_name'])
            sheet.cell(column=2, row=i + 2, value='{:.4f}'.format(stats['random_error']))
            sheet.cell(column=3, row=i + 2, value='{:.4f}'.format(stats['loudnorm_error']))
            sheet.cell(column=4, row=i + 2, value='{:.4f}'.format(stats['mix_error']))
            random_errors.append(stats['random_error'])
            loudnorm_errors.append(stats['loudnorm_error'])
            mix_errors.append(stats['mix_error'])

        final_row = len(songlist) + 2
        sheet.cell(column=1, row=final_row, value='Mean')
        sheet.cell(column=2, row=final_row, value='{:.4f}'.format(mean(random_errors)))
        sheet.cell(column=3, row=final_row, value='{:.4f}'.format(mean(loudnorm_errors)))
        sheet.cell(column=4, row=final_row, value='{:.4f}'.format(mean(mix_errors)))

        book.save('./stats.xlsx')


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Torch version: ', torch.__version__)
    print('Device: ', device)

    base_path = '/media/apelykh/bottomless-pit/datasets/mixing/MedleyDB/Audio'
    chunk_length = 1

    train_songlist = ['Patrick Talbot - A Reason To Leave', 'Traffic Experiment - Once More (With Feeling)',
                      'CelestialShore_DieForUs', 'Titanium - Haunted Age', 'James May - If You Say',
                      'Fergessen - Nos Palpitants', "The Wrong'Uns - Rothko", 'Hollow Ground - Left Blind',
                      'Atlantis Bound - It Was My Fault For Waiting', 'Jokers, Jacks & Kings - Sea Of Leaves',
                      'Cnoc An Tursa - Bannockburn', 'James May - All Souls Moon', 'Fergessen - The Wind',
                      'TheSoSoGlos_Emergency', 'Johnny Lokke - Whisper To A Scream', 'StevenClark_Bounty',
                      'James May - On The Line', 'PortStWillow_StayEven', 'Voelund - Comfort Lives In Belief',
                      'TheDistricts_Vermont', 'Leaf - Summerghost', 'FacesOnFilm_WaitingForGa',
                      'ANiMAL - Easy Tiger', 'Skelpolu - Together Alone', 'Actions - One Minute Smile',
                      'Flags - 54', 'Angela Thomas Wade - Milk Cow Blues', 'InvisibleFamiliars_DisturbingWildlife',
                      'PurlingHiss_Lolita', 'North To Alaska - All The Same', 'HeladoNegro_MitadDelMundo',
                      'Swinging Steaks - Lost My Way', 'Creepoid_OldTree', 'StrandOfOaks_Spacestation',
                      "Phre The Eon - Everybody's Falling Apart", 'Grants - PunchDrunk', 'AClassicEducation_NightOwl',
                      'ANiMAL - Clinic A', 'The Long Wait - Back Home To Blue', 'NightPanther_Fire',
                      'Young Griffo - Pennies', 'Black Bloc - If You Want Success', 'Tim Taler - Stalker',
                      'SecretMountains_HighHorse', 'HezekiahJones_BorrowedHeart', 'Young Griffo - Blood To Bone',
                      'Lushlife_ToynbeeSuite', 'AvaLuna_Waterduct', 'James May - Dont Let Go', 'Grants_PunchDrunk',
                      'Triviul - Angelsaint', 'Snowmine_Curfews', 'Leaf - Come Around',
                      'Johnny Lokke - Promises & Lies', 'Bill Chudziak - Children Of No-one', 'Triviul - Dorothy',
                      'Patrick Talbot - Set Me Free', 'Jay Menon - Through My Eyes', 'Drumtracks - Ghost Bitch',
                      'ANiMAL - Rockshow', "Actions - Devil's Words", 'St Vitus - Word Gets Around',
                      'Remember December - C U Next Time', 'Dark Ride - Burning Bridges']

    val_songlist = ['Auctioneer_OurFutureFaces', 'FamilyBand_Again', 'DreamersOfTheGhetto_HeavyLove',
                    'Skelpolu - Human Mistakes', 'HopAlong_SisterCities', 'Actions - South Of The Water',
                    'Wall Of Death - Femme', 'Young Griffo - Facade', 'Giselle - Moss', 'Leaf - Wicked',
                    'Fergessen - Back From The Start', 'Chris Durban - Celebrate', 'BigTroubles_Phantom',
                    'Traffic Experiment - Sirens', "Spike Mullings - Mike's Sulking", 'SweetLights_YouLetMeDown']

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
    weights = './saved_models/from_server/21-08-2020-10:49_training_4masks_unnorm_1s_medleydb+musdb_train_30_epochs_110_val_loss/scalar1s_23_neg_train_loss=-109.7035.pt'
    model.load_state_dict(torch.load(weights, map_location=device))

    base_dir = '/media/apelykh/bottomless-pit/datasets/mixing/MUSDB18HQ/test/'
    # song_name = 'Arise - Run Run Run'
    song_name = "BKS - Bulldozer"

    l_eval = LoudnessEvaluator(d, mean_loudness, model)
    stats = l_eval.process_song(base_dir, song_name, write_wavs_to_disk=True)
    print(json.dumps(stats, indent=4))
    # l_eval.process_songlist(base_dir, songlist)

    # l_eval.evaluate_medleyb_song(loaded_tracks, song_name)

