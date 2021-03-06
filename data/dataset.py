import os
import random
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import librosa
import time
import torch
import torchaudio.functional
from torch.utils import data
from statistics import mean

# TODO: switch sequential sampling order to random sampling


class MultitrackAudioDataset(data.Dataset):
    def __init__(self, base_path: str, songlist: list = None, chunk_length: int = 5,
                 sr: int = 44100, seed: int = None, normalize: bool = False,
                 compute_features: bool = True, augment_data: bool = False):
        """
        Dataset with song caching for quicker load. A song stays in cache until the
        features were not computed for all its chunks.
        Consequences:
        - although a song list can be shuffled, chunks from each individual song
            will be yielded sequentially;

        :param base_path: path to the data folder root;
        :param songlist: if set, only songs from the list will be loaded, otherwise,
            all songs from the root folder;
        :param chunk_length: length (in seconds) of the audio chunk to compute features for;
        :param sr: sampling rate of data samples;
        :param seed: random seed to fix the way data is shuffled;
        :param normalize: if True, audio and spectrograms are normalized to the range of [-1, 1];
        TODO: update
        """
        self._base_path = base_path
        self._chunk_length = chunk_length
        self._normalize = normalize
        self._compute_features = compute_features
        self._augment = augment_data
        self._sr = sr
        self._tracklist = ['bass', 'drums', 'vocals', 'other', 'mix']

        if not songlist:
            songlist = [song_name for song_name in os.listdir(self._base_path) if
                        os.path.isdir(os.path.join(self._base_path, song_name))]

        self.songlist = songlist

        if seed:
            random.seed(seed)
        random.shuffle(songlist)

        self._len, self.song_durations = self._calculate_dataset_length()

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
            song_path = os.path.join(self._base_path, song_name)
            track_path = os.path.join(song_path, '{}_MIX.wav'.format(song_name))
            song_duration = int(sf.info(track_path).duration)
            # trimmed song duration in seconds
            song_durations.append(song_duration - (song_duration % self._chunk_length))
            dataset_len += int(song_duration / self._chunk_length)

        return dataset_len, song_durations

    def _get_track_path(self, song_name, track_name):
        song_path = os.path.join(self._base_path, song_name)

        if track_name == 'mix':
            track_path = os.path.join(song_path, '{}_MIX.wav'.format(song_name))
        else:
            track_path = os.path.join(song_path, '{}_STEMS_JOINED'.format(song_name),
                                      '{}_STEM_{}.wav'.format(song_name, track_name.upper()))
        return track_path

    # # TODO: stop using MUSDB18 for training -> remove the support for file naming?
    # def _get_musdb18_track_path(self, song_name, track_name):
    #     song_path = os.path.join(self._base_path, song_name)
    #
    #     if track_name == 'mix':
    #         track_path = os.path.join(song_path, 'mixture.wav')
    #     else:
    #         track_path = os.path.join(song_path, '{}.wav'.format(track_name))
    #     return track_path

    def _calculate_song_index(self, chunk_i: int) -> tuple:
        """
        Calculate an index of the song in a song list, based on the chunk index.

        :param chunk_i: global index of the song chunk in range [0, len(dataset)];
        :return: [0]: index of the song in a song list, the chunk is taken from;
                 [1]: index of a chunk in the song;
        """
        song_i = 0
        num_chunks_in_song = int(self.song_durations[song_i] / self._chunk_length)

        while chunk_i >= num_chunks_in_song and song_i < len(self.songlist) - 1:
            chunk_i -= num_chunks_in_song
            song_i += 1
            num_chunks_in_song = int(self.song_durations[song_i] / self._chunk_length)

        return song_i, chunk_i

    def compute_mean_loudness(self) -> dict:
        print('[.] Computing mean loudness...')
        loudness = {track_name: [] for track_name in self._tracklist}
        meter = pyln.Meter(self._sr)

        for song_i, song_name in enumerate(self.songlist):
            print('{}/{}: {}'.format(song_i + 1, len(self.songlist), song_name))

            for track_name in self._tracklist:
                track_path = self._get_track_path(song_name, track_name)
                track, _ = sf.read(track_path)
                track_loudness = meter.integrated_loudness(track)
                loudness[track_name].append(track_loudness)

        mean_loudness = {track_name: mean(loudness[track_name]) for track_name in loudness}
        return mean_loudness

    def compute_features(self, audio: np.ndarray, window_size: int = 2048,
                         hop_length: int = 1024):
        """
        Compute STFT features for an input audio.
        """
        # spectrum = librosa.stft(audio,
        #                         n_fft=window_size,
        #                         hop_length=hop_length,
        #                         window=np.hanning(window_size))
        # magnitudes = np.abs(spectrum)
        # features = librosa.amplitude_to_db(magnitudes)

        # -------------------------------------------------------------------------
        spectrum = torch.stft(torch.from_numpy(audio),
                              n_fft=window_size,
                              hop_length=hop_length,
                              window=torch.hann_window(window_size),
                              return_complex=True)

        magnitudes = torch.abs(spectrum)
        features = torchaudio.functional.amplitude_to_DB(magnitudes,
                                                         multiplier=20,
                                                         amin=1e-5,
                                                         db_multiplier=0)
        # -------------------------------------------------------------------------
        # features = magnitudes

        # if self._normalize:
        #     features = librosa.util.normalize(features)

        return features

    @staticmethod
    def _augment_audio(audio: np.array, gain_from: float = 0.6,
                       gain_to: float = 1.4) -> np.array:
        random_gain = np.random.uniform(gain_from, gain_to)
        return random_gain * audio

    @staticmethod
    def _augment_features(features: np.ndarray, gain_from: float = 0.6,
                          gain_to: float = 1.4) -> np.ndarray:
        random.seed(None)
        rangom_gains = np.random.uniform(gain_from, gain_to, size=len(features))
        gains_db = 20 * np.log10(rangom_gains)
        # np.newaxis added to broadcast the shape
        augm_features = np.add(features, gains_db[:, np.newaxis, np.newaxis])

        return augm_features

    @staticmethod
    def _stereo_to_mono(audio: np.ndarray) -> np.array:
        return np.mean(audio, axis=1)

    def _process_on_the_fly(self, song_i: int, chunk_i: int) -> tuple:
        song_name = self.songlist[song_i]
        sample_from = chunk_i * self._chunk_length * self._sr
        sample_to = (chunk_i + 1) * self._chunk_length * self._sr
        per_track_features = []
        gt_features = None

        for track_name in self._tracklist:
            track_path = self._get_track_path(song_name, track_name)
            audio_chunk, _ = sf.read(track_path, start=sample_from, stop=sample_to)
            if audio_chunk.shape[1] == 2:
                audio_chunk = self._stereo_to_mono(audio_chunk)

            if self._augment:
                audio_chunk = MultitrackAudioDataset._augment_audio(audio_chunk)

            features = self.compute_features(audio_chunk)
            if track_name == 'mix':
                gt_features = features
            else:
                per_track_features.append(features)

        train_features = torch.stack(per_track_features)
        # train_features = np.stack(per_track_features)

        return train_features, gt_features

    # TODO: update
    def _precompute_features(self):
        """
        Pre-compute features and save them to disk as .npy files.
        """
        print('[.] Pre-computing features...')

        for song_i, song_name in enumerate(self.songlist):
            print('-' * 60)
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
                suffix = '_norm' if self._normalize else ''

                for track in self._tracklist:
                    audio_chunk = self._loaded_track[track][i_from:i_to]
                    features = self.compute_features(audio_chunk)

                    if track == 'mix':
                        np.save(os.path.join(features_dir, '{}_gt_{}s{}.npy'.
                                             format(chunk_i, self._chunk_length, suffix)), features)
                    else:
                        per_track_features.append(features)

                np.save(os.path.join(features_dir, '{}_train_{}s{}.npy'.format(chunk_i, self._chunk_length, suffix)),
                        np.stack(per_track_features))
        print('[+] Features computed and saved')

    def _process_precomputed(self, song_i, chunk_i) -> tuple:
        """
        Read pre-computed features from drive.
        """
        song_name = self.songlist[song_i]
        song_dir = os.path.join(self._base_path, song_name)
        features_dir = os.path.join(song_dir, '{}_FEATURES'.format(song_name))

        suffix = '_norm' if self._normalize else ''
        gt_features = np.load(os.path.join(features_dir, '{}_gt{}.npy'.format(chunk_i, suffix)))
        train_features = np.load(os.path.join(features_dir, '{}_train{}.npy'.format(chunk_i, suffix)))

        if self._augment:
            train_features = MultitrackAudioDataset._augment_features(train_features)

        return train_features, gt_features

    def __getitem__(self, index: int) -> tuple:
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
        print('Song {}, chunk {}'.format(self.songlist[song_i], chunk_i))

        if self._compute_features:
            tic = time.time()
            train_features, gt_features = self._process_on_the_fly(song_i, chunk_i)
            print('Features: {}'.format(time.time() - tic))
            return train_features, gt_features
        else:
            return self._process_precomputed(song_i, chunk_i)

    def __len__(self) -> int:
        return self._len

    def get_num_songs(self) -> int:
        return len(self.songlist)

    def get_song_durations(self) -> list:
        return self.song_durations

    def get_tracklist(self) -> list:
        return self._tracklist


if __name__ == '__main__':
    import time

    tic = time.time()
    d = MultitrackAudioDataset(base_path='/media/apelykh/bottomless-pit/datasets/mixing/MedleyDB/Audio',
                               chunk_length=5,
                               normalize=False,
                               compute_features=True,
                               augment_data=False)
    print('Dataset loaded in {}'.format(time.time() - tic))
    print(len(d))
    tic = time.time()
    for i in range(100):
        sample = d[i]
        if i % 20 == 0:
            print('{}/{}'.format(i, len(d)))
    print('100 samples loaded in {}'.format(time.time() - tic))

    # tic = time.time()
    # mean_loudness = d.compute_mean_loudness()
    # print('mean loudness computed in {}'.format(time.time() - tic))
