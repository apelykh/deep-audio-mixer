import random
import os
import numpy as np
import librosa
from torch.utils import data


class AudioMixingDataset(data.Dataset):
    def __init__(self, base_path: str, chunk_length: int = 5, train_val_test_split: tuple = (1.0, 0.0, 0.0),
                 mode: str = 'train', seed: int = None, normalize: bool = False):
        """
        :param base_path: path to the data folder;
        :param chunk_length: length (in seconds) of the audio chunk to compute features for;
        :param train_val_test_split: fraction of the data, reserved for the
            train/val/test set correspondingly;
        :param mode: one of ["train", "val", "test"];
        :param seed: random seed to fix the way data is shuffled;
        :param normalize: if True, audio is normalized to the range of [-1, 1];
        """
        self._base_path = base_path
        self._chunk_length = chunk_length
        self._normalize = normalize

        self.sr = 44100
        self._tracklist = ['accompaniment', 'bass', 'drums', 'vocals', 'other', 'mixture']
        self._track_mask = None
        self._loaded_track_i = None
        self._loaded_track = {}
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

        self.song_durations = []
        self._len = self._calculate_dataset_length()

    def _calculate_dataset_length(self) -> int:
        dataset_len = 0
        for song_name in self.songlist:
            track_name = 'mixture'
            track_path = os.path.join(self._base_path, song_name, '{}.wav'.format(track_name))
            song_duration = librosa.get_duration(filename=track_path, sr=self.sr)
            # self.song_durations.append(song_duration)
            self.song_durations.append(song_duration - (song_duration % self._chunk_length))

            dataset_len += int(song_duration / self._chunk_length)
        return dataset_len

    def _load_tracks(self, song_i: int):
        """
        Cache a song not to read if from disk every time. Should be kept in cache until all its chunks
        are used.
        :param song_i:
        """
        print('[+] Loading track {}'.format(song_i))

        self._loaded_track_i = song_i
        song_name = self.songlist[song_i]

        for track_name in self._tracklist:
            track_path = os.path.join(self._base_path, song_name, '{}.wav'.format(track_name))
            self._loaded_track[track_name], _ = librosa.load(track_path, sr=self.sr)
            # trim the track length to be a multiple of self._chunk_length
            len_samples = int(self.song_durations[song_i] * self.sr)
            self._loaded_track[track_name] = self._loaded_track[track_name][:len_samples]

            if self._normalize:
                self._loaded_track[track_name] = librosa.util.normalize(self._loaded_track[track_name])

        num_chunks = int(self.song_durations[song_i] / self._chunk_length)
        # mask will show which track chunks were already used
        self._track_mask = np.ones(num_chunks, dtype=np.int8)

    def _unload_tracks(self):
        print('[-] Unloading track {}'.format(self._loaded_track_i))

        self._loaded_track = {}
        self._track_mask = None
        self._loaded_track_i = None

    def _calculate_song_index(self, chunk_i: int) -> int:
        song_i = 0
        # number of chunks in the first song
        max_chunk_i = int(self.song_durations[0] / self._chunk_length) - 1

        # TODO: fix issue with song index for the last song
        while chunk_i > max_chunk_i and song_i < len(self.songlist) - 1:
            song_i += 1
            max_chunk_i += int(self.song_durations[song_i] / self._chunk_length)

        return song_i

    def __getitem__(self, index: int) -> dict:
        song_i = self._calculate_song_index(index)
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
        print(free_chunks)
        print('Num free chunks: ', len(free_chunks))
        # good idea?
        random.seed(None)
        chunk_i = np.random.choice(free_chunks)
        result['chunk_index'] = chunk_i
        print('Chunk index: ', chunk_i)

        i_from = chunk_i * self._chunk_length * self.sr
        i_to = (chunk_i + 1) * self._chunk_length * self.sr
        for track in self._tracklist:
            audio_chunk = self._loaded_track[track][i_from:i_to]
            result['{}_audio'.format(track)] = audio_chunk
            feature = librosa.feature.melspectrogram(audio_chunk, sr=self.sr, n_fft=2048, hop_length=1024)
            result['{}_feature'.format(track)] = feature

        self._track_mask[chunk_i] = 0
        # print(self._track_mask)

        if np.sum(self._track_mask) == 0:
            self._unload_tracks()

        return result

    def get_num_songs(self) -> int:
        return len(self.songlist)

    def __len__(self) -> int:
        return self._len


if __name__ == '__main__':
    d = AudioMixingDataset('/media/apelykh/bottomless-pit/datasets/mixing/MUSDB18HQ/test',
                           chunk_length=5, train_val_test_split=(0.0, 0.2, 0.8),
                           mode='val', seed=123)
    print(len(d))
    d[47]['song_index']
    d[86]['song_index']
    d[88]['song_index']
