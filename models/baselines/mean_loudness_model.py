import pyloudnorm as pyln


class MeanLoudnessModel:
    def __init__(self, d_mean_loudness: dict, sr=44100):
        self.mean_loudness = d_mean_loudness
        self.meter = pyln.Meter(sr)
        self.tracklist = ('bass', 'drums', 'vocals', 'other')

    def forward(self, x: dict) -> dict:
        result = {}

        for track_name in self.tracklist:
            track_loudness = self.meter.integrated_loudness(x[track_name].T)
            loudness_norm_track = pyln.normalize.loudness(x[track_name].T,
                                                          track_loudness,
                                                          self.mean_loudness[track_name])
            result[track_name] = loudness_norm_track.T

        return result
