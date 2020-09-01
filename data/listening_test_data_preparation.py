import os
import numpy as np
# import librosa
import pyloudnorm as pyln
import soundfile as sf
import pickle
import torch
from data.dataset import MultitrackAudioDataset
from data.dataset_utils import load_tracks_musdb18
# from models.model_scalar_1s import MixingModelScalar1s
from models.model_scalar_2s import MixingModelScalar2s
from models.baselines.random_model import RandomModel
from models.baselines.mean_loudness_model import MeanLoudnessModel
from inference_utils import mix_song_smooth

meter = pyln.Meter(44100)


def produce_mixture_and_save(track_dict: dict, song_name, identifier, save_dir, sr=44100):
    track_arr = np.array(list(track_dict.values()))
    track_sum = np.sum(track_arr, axis=0)

    loudness = meter.integrated_loudness(track_sum.T)
    loudness_norm_sum = pyln.normalize.loudness(track_sum.T, loudness, -20.0)
    sf.write(os.path.join(save_dir, '{}_{}.wav'.format(song_name, identifier)), loudness_norm_sum, sr)


def process_song(base_dir: str, song_name: str, time_interval: tuple,
                 models: dict, dataset, save_dir, sr: int = 44100):
    sample_from = time_interval[0] * sr
    sample_to = time_interval[1] * sr

    # save reference mix
    loaded_tracks = load_tracks_musdb18(os.path.join(base_dir, 'manual_gain_mixes'), song_name,
                                        tracklist=('bass', 'drums', 'vocals', 'other'))
    loaded_tracks = {track_name: track[:, sample_from:sample_to]
                     for track_name, track in loaded_tracks.items()}
    produce_mixture_and_save(loaded_tracks, song_name, 'reference', save_dir)

    # save raw sum
    loaded_tracks = load_tracks_musdb18(os.path.join(base_dir, 'test'), song_name,
                                        tracklist=('bass', 'drums', 'vocals', 'other'))
    loaded_tracks = {track_name: track[:, sample_from:sample_to]
                     for track_name, track in loaded_tracks.items()}
    produce_mixture_and_save(loaded_tracks, song_name, 'sum', save_dir)

    for model_name in models:
        if model_name == 'mix':
            mixed_tracks, _, _ = mix_song_smooth(dataset, models[model_name],
                                                 loaded_tracks, chunk_length=2)
        else:
            mixed_tracks = models[model_name].forward(loaded_tracks)
        produce_mixture_and_save(mixed_tracks, song_name, model_name, save_dir)


def process_songlist(base_dir, songlist, time_intervals,
                     models, dataset, save_dir='./test_data'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, song_name in enumerate(songlist):
        print('{}/{}: {}'.format(i + 1, len(songlist), song_name))
        time_interval = time_intervals[i]
        process_song(base_dir, song_name, time_interval, models, dataset, save_dir)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)

    chunk_length = 2

    # for 2s model
    train_songlist = ['Actions - One Minute Smile', 'ANiMAL - Clinic A', "Actions - Devil's Words",
                      'Young Griffo - Blood To Bone',
                      'Jokers, Jacks & Kings - Sea Of Leaves', 'Auctioneer_OurFutureFaces', 'Grants_PunchDrunk',
                      'Giselle - Moss',
                      'The Long Wait - Back Home To Blue', 'Remember December - C U Next Time',
                      'Johnny Lokke - Whisper To A Scream',
                      "The Wrong'Uns - Rothko", 'Chris Durban - Celebrate', 'St Vitus - Word Gets Around',
                      'AClassicEducation_NightOwl', 'Grants - PunchDrunk', 'InvisibleFamiliars_DisturbingWildlife',
                      'Skelpolu - Human Mistakes', 'PurlingHiss_Lolita', 'Jay Menon - Through My Eyes',
                      'James May - On The Line',
                      'FamilyBand_Again', 'StevenClark_Bounty', 'HeladoNegro_MitadDelMundo',
                      'DreamersOfTheGhetto_HeavyLove',
                      'Fergessen - The Wind', 'James May - All Souls Moon', 'Bill Chudziak - Children Of No-one',
                      'BigTroubles_Phantom', 'Dark Ride - Burning Bridges', 'Fergessen - Nos Palpitants',
                      'North To Alaska - All The Same', 'PortStWillow_StayEven', 'SweetLights_YouLetMeDown',
                      'ANiMAL - Easy Tiger',
                      'Leaf - Summerghost', 'HezekiahJones_BorrowedHeart', 'Hollow Ground - Left Blind',
                      'Johnny Lokke - Promises & Lies', 'Atlantis Bound - It Was My Fault For Waiting',
                      'Voelund - Comfort Lives In Belief', 'Swinging Steaks - Lost My Way', 'Young Griffo - Facade',
                      'Titanium - Haunted Age', 'Traffic Experiment - Once More (With Feeling)',
                      "Phre The Eon - Everybody's Falling Apart", 'Black Bloc - If You Want Success',
                      'Angela Thomas Wade - Milk Cow Blues', 'Flags - 54', 'Patrick Talbot - A Reason To Leave',
                      'TheDistricts_Vermont', 'Leaf - Wicked', 'Creepoid_OldTree', 'HopAlong_SisterCities',
                      'AvaLuna_Waterduct',
                      'SecretMountains_HighHorse', 'Drumtracks - Ghost Bitch', 'Cnoc An Tursa - Bannockburn',
                      'Patrick Talbot - Set Me Free', 'Triviul - Angelsaint', 'FacesOnFilm_WaitingForGa',
                      'Triviul - Dorothy',
                      'Skelpolu - Together Alone', 'Actions - South Of The Water']

    d = MultitrackAudioDataset(
        '/media/apelykh/bottomless-pit/datasets/mixing/MedleyDB/Audio',
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

    base_dir = '/media/apelykh/bottomless-pit/datasets/mixing/MUSDB18HQ'
    test_songlist = [
        "Arise - Run Run Run",
        "BKS - Bulldozer",
        "BKS - Too Much",
        "Bobby Nobody - Stitch Up",
        "Cristina Vane - So Easy",
        "Enda Reilly - Cur An Long Ag Seol",
        "Forkupines - Semantics",
        "James Elder & Mark M Thompson - The English Actor",
        "Nerve 9 - Pray For The Rain",
        "Raft Monk - Tiring",
        "Signe Jakobsen - What Have You Done To Me",
        "Speak Softly - Broken Man",
        "The Doppler Shift - Atrophy",
        "Timboz - Pony",
        "Zeno - Signs"
    ]
    time_intervals = [
        (80, 80 + 30),
        (25, 25 + 30),
        (35, 35 + 30),
        (65, 65 + 30),
        (60, 60 + 30),
        (80, 80 + 30),
        (150, 150 + 30),
        (50, 50 + 30),
        (41, 41 + 30),
        (41, 41 + 30),
        (41, 41 + 30),
        (28, 28 + 30),
        (60, 60 + 30),
        (196, 196 + 30),
        (43, 43 + 30)
    ]

    mixing_model = MixingModelScalar2s().to(device)
    weights = '../saved_models/from_server/21-08-2020-16:44_training_4masks_unnorm_2s_medleydb+musdb_train_26_epochs_168.41_val_loss/best_scalar2s_19_neg_train_loss=-168.4133.pt'
    mixing_model.load_state_dict(torch.load(weights, map_location=device))

    models = {
        'random': RandomModel(),
        'loudnorm': MeanLoudnessModel(mean_loudness),
        'mix': mixing_model,
    }

    process_songlist(base_dir, test_songlist, time_intervals, models, d)
