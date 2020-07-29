import os
import random
import numpy as np
import librosa
import torch
from torch.utils import data


def no_phase_collate_fn(batch):
    target_keys = ['gt_features', 'train_features', 'sum_features']
    batch_dict = {}

    for elem in batch:
        for key in target_keys:
            if key not in batch_dict:
                batch_dict[key] = []
            batch_dict[key].append(elem[key])

    return {key: torch.tensor(batch_dict[key]) for key in batch_dict}


class MultitrackAudioDataset(data.Dataset):
    def __init__(self, base_path: str, songlist: list = None, chunk_length: int = 1,
                 sr: int = 44100, train_val_test_split: tuple = (1.0, 0.0, 0.0),
                 mode: str = 'train', seed: int = None, normalize: bool = True,
                 compute_features: bool = False):
        """
        Dataset with song caching for quicker load. A song stays in cache until the
        features were not computed for all its chunks.
        Consequences:
        - probably won't work with multiprocessing so don't set multiple workers in DataLoader :(
        - although a song list can be shuffled, chunks from each individual song
            will be yielded sequentially;

        :param base_path: path to the data folder root;
        :param songlist: if set, only songs from the list will be loaded, otherwise,
            all songs from the root folder;
        :param chunk_length: length (in seconds) of the audio chunk to compute features for;
        :param sr: sampling rate of data samples;
        :param train_val_test_split: fraction of the data, reserved for
            train/val/test set correspondingly;
        :param mode: one of ["train", "val", "test"];
        :param seed: random seed to fix the way data is shuffled;
        :param normalize: if True, audio and spectrograms are normalized to the range of [-1, 1];
        """
        self._base_path = base_path
        self._chunk_length = chunk_length
        self._normalize = normalize
        self._compute_features = compute_features
        self._sr = sr
        self._tracklist = ['bass', 'drums', 'vocals', 'other', 'mix']
        self._track_mask = None
        self._loaded_track_i = None
        self._loaded_track = {}

        if songlist:
            self.songlist = songlist
        else:
            self.songlist = [song_name for song_name in os.listdir(self._base_path) if
                             os.path.isdir(os.path.join(self._base_path, song_name))]
        if seed:
            random.seed(seed)
        random.shuffle(self.songlist)

        self.songlist = self._split_songlist(mode, train_val_test_split)

        self._len, self.song_durations = self._calculate_dataset_length()

    def _split_songlist(self, mode, train_val_test_split) -> list:
        """
        Select a part of song list for the current dataset instance depending on the mode.
        """
        if mode == 'train':
            songlist = self.songlist[:round(len(self.songlist) * train_val_test_split[0])]
        elif mode == 'val':
            i_from = round(len(self.songlist) * train_val_test_split[0])
            i_to = round(len(self.songlist) * (train_val_test_split[0] + train_val_test_split[1]))
            songlist = self.songlist[i_from:i_to]
        elif mode == 'test':
            songlist = self.songlist[-round(len(self.songlist) * train_val_test_split[2]):]
        else:
            raise ValueError("Mode must be one of train/val/test")

        return songlist

    def _load_tracks(self, song_i: int):
        """
        Cache a song to avoid reading if from disk every time.
        Should be kept in cache until all its chunks are used.

        :param song_i: index of the song to cache;
        """
        self._loaded_track_i = song_i
        song_name = self.songlist[song_i]
        song_path = os.path.join(self._base_path, song_name)

        for track_name in self._tracklist:
            if track_name == 'mix':
                track_path = os.path.join(song_path, '{}_MIX.wav'.format(song_name))
            else:
                track_path = os.path.join(song_path, '{}_STEMS_JOINED'.format(song_name),
                                          '{}_STEM_{}.wav'.format(song_name, track_name.upper()))

            track, _ = librosa.load(track_path, sr=self._sr)
            # trim the track length to be a multiple of self._chunk_length
            len_samples = int(self.song_durations[song_i] * self._sr)
            # peak normalization? does not influence STFT
            self._loaded_track[track_name] = librosa.util.normalize(track[:len_samples])

        num_chunks = int(self.song_durations[song_i] / self._chunk_length)
        # mask will show which track chunks were already used
        self._track_mask = np.ones(num_chunks, dtype=np.int8)

    def _unload_tracks(self):
        """
        Reset the song cache
        """
        self._loaded_track = {}
        self._track_mask = None
        self._loaded_track_i = None

    def _calculate_dataset_length(self) -> tuple:
        """
        Calculate dataset length based on its song list. Length is measured
        in number of chunks of self._chunk_length.

        Note: each song in the dataset is effectively cut to the maximum length,
        proportional to self._chunk_length
        """
        dataset_len = 0
        song_durations = []

        for song_name in self.songlist:
            track_path = os.path.join(self._base_path, song_name, '{}_MIX.wav'.format(song_name))
            song_duration = librosa.get_duration(filename=track_path, sr=self._sr)
            # trimmed song duration in seconds
            song_durations.append(song_duration - (song_duration % self._chunk_length))

            dataset_len += int(song_duration / self._chunk_length)
        return dataset_len, song_durations

    def _calculate_song_index(self, chunk_i: int) -> tuple:
        """
        Calculate an index of the song in a song list, based on the chunk index.

        :param chunk_i: global index of the song chunk in range [0, len(dataset)];
        TODO: update
        :return: index of the song in a song list, the chunk is taken from;
        """
        song_i = 0

        while int(self.song_durations[song_i] / self._chunk_length) <= chunk_i and \
                song_i < len(self.songlist) - 1:
            chunk_i -= int(self.song_durations[song_i] / self._chunk_length)
            song_i += 1

        return song_i, chunk_i

    @staticmethod
    def compute_features(audio: np.ndarray, window_size: int = 2048, hop_length: int = 512) -> tuple:
        """
        Compute STFT features for an input audio.

        :param audio: input audio;
        :param window_size:
        :param hop_length:
        :return:
        """
        spectrum = librosa.stft(audio,
                                n_fft=window_size,
                                hop_length=hop_length,
                                window=np.hanning(window_size))
        magnitudes = np.abs(spectrum)
        # phases = spectrum / magnitudes
        phases = None

        features = librosa.amplitude_to_db(magnitudes)

        # if self._normalize:
        #     features = librosa.util.normalize(features)

        return features, phases

    def _process_on_the_fly(self, song_i):
        """
        The song is cached until all of its chunks are used. On each iteration the chunk
        is chosen randomly among the free ones.
        """
        result = {
            'song_name': self.songlist[song_i],
            'song_index': song_i
        }

        if not self._loaded_track or self._loaded_track_i != song_i:
            self._unload_tracks()
            # cache the song not to load it every time
            self._load_tracks(song_i)

        free_chunks = np.nonzero(self._track_mask)[0]
        random.seed(None)
        # index of a chunk in a current song
        chunk_i = np.random.choice(free_chunks)
        result['chunk_index'] = chunk_i

        # sample indices
        i_from = chunk_i * self._chunk_length * self._sr
        i_to = (chunk_i + 1) * self._chunk_length * self._sr
        sum_chunk = np.zeros(i_to - i_from)
        per_track_features = []

        for track in self._tracklist:
            audio_chunk = self._loaded_track[track][i_from:i_to]
            features, _ = self.compute_features(audio_chunk)

            if track == 'mix':
                result['gt_features'] = features
            else:
                sum_chunk += audio_chunk
                per_track_features.append(features)

        result['train_features'] = np.stack(per_track_features)
        # result['sum_features'], result['sum_phases'] = self.compute_features(sum_chunk)

        # indicate that the current song chunk was used
        self._track_mask[chunk_i] = 0
        # if it was the last chunk of the current song, unload it
        # to load a new one on the next iteration
        if np.sum(self._track_mask) == 0:
            self._unload_tracks()

        return result

    def _precompute_features(self):
        """
        Pre-compute features and save them to disk as .npy files.
        """
        print('[.] Pre-computing features...')

        for song_i, song_name in enumerate(self.songlist):
            print('{}/{}: {}'.format(song_i, len(self.songlist), song_name))
            self._load_tracks(song_i)

            song_dir = os.path.join(self._base_path, song_name)
            features_dir = os.path.join(song_dir, '{}_FEATURES'.format(song_name))

            if not os.path.isdir(features_dir):
                os.makedirs(features_dir)

            num_chunks = int(self.song_durations[song_i] / self._chunk_length)
            for chunk_i in range(num_chunks):
                if chunk_i % 30 == 0:
                    print('{}/{}'.format(chunk_i, num_chunks))

                i_from = chunk_i * self._chunk_length * self._sr
                i_to = (chunk_i + 1) * self._chunk_length * self._sr
                per_track_features = []

                for track in self._tracklist:
                    audio_chunk = self._loaded_track[track][i_from:i_to]
                    features, _ = self.compute_features(audio_chunk)

                    if track == 'mix':
                        np.save(os.path.join(features_dir, '{}_gt.npy'.format(chunk_i)), features)
                    else:
                        per_track_features.append(features)

                np.save(os.path.join(features_dir, '{}_train.npy'.format(chunk_i)),
                        np.stack(per_track_features))
        print('[+] Features computed and saved')

    def _process_precomputed(self, song_i, chunk_i) -> dict:
        """
        Read pre-computed features from drive.
        """
        song_name = self.songlist[song_i]
        result = {
            'song_name': song_name,
            'song_index': song_i,
            'chunk_index': chunk_i
        }

        song_dir = os.path.join(self._base_path, song_name)
        features_dir = os.path.join(song_dir, '{}_FEATURES'.format(song_name))
        result['train_features'] = np.load(os.path.join(features_dir, '{}_train.npy'.format(chunk_i)))
        result['gt_features'] = np.load(os.path.join(features_dir, '{}_gt.npy'.format(chunk_i)))

        return result

    def __getitem__(self, index: int) -> dict:
        """
        :param index: global index of a song chunk to compute features for, in range [0, len(dataset)];
        :return: {
            'song_name': str, song name;
            'song_index': int, index of a song in a songlist the current chunk is taken from;
            'chunk_index': int, index of a chunk in a current song;
            'gt_features': numpy array of shape [feature_high, feature_len],
                features of a chunk from the mixture track;
            'train_features': numpy array of shape [4, feature_high, feature_len],
                stacked features for all stems (4 - bass, drums, other, vocals);
        }
        """
        song_i, chunk_i = self._calculate_song_index(index)
        # print('Song {}, chunk {}'.format(song_i, chunk_i))

        if self._compute_features:
            result = self._process_on_the_fly(song_i)
        else:
            result = self._process_precomputed(song_i, chunk_i)

        return result

    def __len__(self) -> int:
        return self._len

    def get_num_songs(self) -> int:
        return len(self.songlist)

    def get_song_durations(self) -> list:
        return self.song_durations

    def get_tracklist(self) -> list:
        return self._tracklist


if __name__ == '__main__':
    weathervane_music = [
        # 'AClassicEducation_NightOwl',
        # 'Auctioneer_OurFutureFaces',
        # 'AvaLuna_Waterduct',
        # 'BigTroubles_Phantom',
        # 'CelestialShore_DieForUs',
        # 'Lushlife_ToynbeeSuite',
        'NightPanther_Fire',
        'PortStWillow_StayEven',
        'PurlingHiss_Lolita',
        'SecretMountains_HighHorse',
        'Snowmine_Curfews',
        'TheSoSoGlos_Emergency',
        # 'Creepoid_OldTree',
        # 'DreamersOfTheGhetto_HeavyLove',
        # 'FacesOnFilm_WaitingForGa',
        # 'FamilyBand_Again',
        # 'Grants_PunchDrunk',
        # 'HeladoNegro_MitadDelMundo',
        # 'HezekiahJones_BorrowedHeart',
        # 'HopAlong_SisterCities',
        # 'InvisibleFamiliars_DisturbingWildlife',
        # 'StevenClark_Bounty',
        # 'StrandOfOaks_Spacestation',
        # 'SweetLights_YouLetMeDown',
        # 'TheDistricts_Vermont'
    ]

    d_train = MultitrackAudioDataset(
        '/media/apelykh/bottomless-pit/datasets/mixing/MedleyDB/Audio',
        songlist=weathervane_music,
        chunk_length=1,
        train_val_test_split=(0.8, 0.2, 0.0),
        mode='train',
        seed=321,
        normalize=False
    )

    for i in range(len(d_train)):
        sample = d_train[i]
