import numpy as np


class RandomModel:
    def __init__(self, gain_from=0.5, gain_to=1.5):
        self.tracklist = ('bass', 'drums', 'vocals', 'other')
        self._gain_from = gain_from
        self._gain_to = gain_to

    def forward(self, x: dict) -> dict:
        result = dict({track: float(np.random.uniform(self._gain_from, self._gain_to)) * x[track]
                       for track in self.tracklist})

        return result
