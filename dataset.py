import os
import random
import numpy as np
import librosa
from torch.utils import data


class MultitrackAudioDataset(data.Dataset):
    def __init__(self, base_path: str, songlist: list = None, chunk_length: int = 1,
                 sr: int = 44100, train_val_test_split: tuple = (1.0, 0.0, 0.0),
                 mode: str = 'train', seed: int = None, normalize: bool = True,
                 verbose=False):
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
        :param verbose: if True, console logs are displayed;
        """
        self._base_path = base_path
        self._chunk_length = chunk_length
        self._normalize = normalize
        self._verbose = verbose
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

        if mode == 'train':
            self.songlist = self.songlist[:round(len(self.songlist) * train_val_test_split[0])]
        elif mode == 'val':
            i_from = round(len(self.songlist) * train_val_test_split[0])
            i_to = round(len(self.songlist) * (train_val_test_split[0] + train_val_test_split[1]))
            self.songlist = self.songlist[i_from:i_to]
        elif mode == 'test':
            self.songlist = self.songlist[-round(len(self.songlist) * train_val_test_split[2]):]

        # in seconds
        self.song_durations = []
        self._len = self._calculate_dataset_length()

    def _calculate_dataset_length(self) -> int:
        """
        Calculate dataset length based on its songlist. Length is measured
        in number of chunks of self._chunk_length.

        Note: each song in the dataset is effectively cut to the maximum length,
        proportional to self._chunk_length
        """
        dataset_len = 0
        for song_name in self.songlist:
            track_path = os.path.join(self._base_path, song_name, '{}_MIX.wav'.format(song_name))
            song_duration = librosa.get_duration(filename=track_path, sr=self._sr)
            # trimmed song duration
            self.song_durations.append(song_duration - (song_duration % self._chunk_length))

            dataset_len += int(song_duration / self._chunk_length)
        return dataset_len

    def _load_tracks(self, song_i: int):
        """
        Cache a song to avoid reading if from disk every time.
        Should be kept in cache until all its chunks are used.

        :param song_i: index of the song to cache;
        """
        if self._verbose:
            print('[+] Loading track {}'.format(song_i))

        self._loaded_track_i = song_i
        song_name = self.songlist[song_i]
        song_path = os.path.join(self._base_path, song_name)

        for track_name in self._tracklist:
            if track_name == 'mix':
                track_path = os.path.join(song_path, '{}_MIX.wav'.format(song_name))
            else:
                track_path = os.path.join(song_path, '{}_STEMS_JOINED'.format(song_name),
                                          '{}_STEM_{}.wav'.format(song_name, track_name.upper()))

            self._loaded_track[track_name], _ = librosa.load(track_path, sr=self._sr)
            # trim the track length to be a multiple of self._chunk_length
            len_samples = int(self.song_durations[song_i] * self._sr)
            self._loaded_track[track_name] = self._loaded_track[track_name][:len_samples]

            if self._normalize:
                self._loaded_track[track_name] = librosa.util.normalize(self._loaded_track[track_name])

        num_chunks = int(self.song_durations[song_i] / self._chunk_length)
        # mask will show which track chunks were already used
        self._track_mask = np.ones(num_chunks, dtype=np.int8)

    def _unload_tracks(self):
        """
        Reset the song cache
        """
        if self._verbose:
            print('[-] Unloading track {}'.format(self._loaded_track_i))

        self._loaded_track = {}
        self._track_mask = None
        self._loaded_track_i = None

    def _calculate_song_index(self, chunk_i: int) -> int:
        """
        Calculate an index of the song in a song list, based on the chunk index.

        :param chunk_i: index of the song chunk;
        :return: index of the song in a song list, the chunk is taken from;
        """
        song_i = 0
        # number of chunks in the first song
        max_chunk_i = int(self.song_durations[0] / self._chunk_length) - 1

        while chunk_i > max_chunk_i and song_i < len(self.songlist) - 1:
            song_i += 1
            max_chunk_i += int(self.song_durations[song_i] / self._chunk_length)

        return song_i

    def compute_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute features (SFTF in this case) for an input audio.

        :param audio: input audio;
        :return: numpy array with computed features;
        """
        features = librosa.stft(audio, n_fft=2048, hop_length=1024)
        features = librosa.amplitude_to_db(np.abs(features))
        if self._normalize:
            features = librosa.util.normalize(features)

        return features

    def __getitem__(self, index: int) -> dict:
        """
        :param index: index of a song chunk to compute features for;
        :return: {
            'song_name': str, the revelation of the great mysteries of the Universe!
                (kidding, just the song name);
            'song_index': int, index of a song in a songlist the current chunk is taken from;
            'chunk_index': int, index of a chunk in a current song;
            'gt_features': numpy array of shape [feature_high, feature_len],
                features of a chunk from the mixture track;
            'train_features': numpy array of shape [4, feature_high, feature_len],
                stacked features for all stems (4 - bass, drums, other, vocals);
        }
        """
        song_i = self._calculate_song_index(index)
        if self._verbose:
            print('Song index: ', song_i)

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

        if self._verbose:
            print(free_chunks)
            print('Num free chunks: ', len(free_chunks))
            print('Chunk index: ', chunk_i)

        # sample indices
        i_from = chunk_i * self._chunk_length * self._sr
        i_to = (chunk_i + 1) * self._chunk_length * self._sr
        features = []

        for track in self._tracklist:
            audio_chunk = self._loaded_track[track][i_from:i_to]
            feature = self.compute_features(audio_chunk)

            if track != 'mix':
                features.append(feature)
            else:
                result['gt_features'] = feature

        result['train_features'] = np.stack(features)
        # indicate that the current song chunk was used
        self._track_mask[chunk_i] = 0
        # if it was the last chunk of the current song, unload it
        # to load a new one on the next iteration
        if np.sum(self._track_mask) == 0:
            self._unload_tracks()

        return result

    def __len__(self) -> int:
        return self._len

    def get_num_songs(self) -> int:
        return len(self.songlist)

    def get_tracklist(self) -> list:
        return self._tracklist
