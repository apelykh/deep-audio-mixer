import json
import itertools
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
from collections import OrderedDict


def parse_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    scores_by_model = {key: [] for key in ['sum', 'reference', 'mix', 'random', 'loudnorm']}
    scores_by_song = {}

    for page_data in data['pages']:
        song_name = page_data['id']
        print(song_name)
        scores_by_song[song_name] = OrderedDict({})

        for elem in page_data['elements']:
            model_name = elem['id']
            model_id = model_name.split('_')[-1]
            scores = elem['axis'][0]['values']
            print('{} ({} scores): {}'.format(model_id, len(scores), scores))

            scores_by_song[song_name][model_id] = scores
            scores_by_model[model_id].append(scores)

    return scores_by_model, scores_by_song


def produce_boxplot(data, keys, plot_name):
    fig = plt.figure(figsize=(7, 5))
    # Creating axes instance
    # ax = fig.add_axes([0, 0, 1, 1])

    medianprops = dict(linestyle='-', linewidth=3.0, color='orange')
    bp_dict = plt.boxplot(data,
                patch_artist=True,
                medianprops=medianprops)

    for line in bp_dict['medians']:
        # get position data for median line
        x, y = line.get_xydata()[1]  # top of median line
        # overlay median value
        text(x, y, '%.2f' % y,
             horizontalalignment='left')  # draw above, centered

    xtick_locs = [elem + 1 for elem in range(len(keys))]
    keys = [key if key != 'mix' else 'CNN' for key in sorted(global_scores.keys())]
    plt.xticks(xtick_locs, keys)
    fig.savefig('./test_figures/{}.png'.format(plot_name))
    plt.close(fig)
    # plt.show()


if __name__ == '__main__':
    json_path = '/home/apelykh/Downloads/scores_no_zero.json'

    scores_by_model, song_scores = parse_json(json_path)

    global_scores = {key: list(itertools.chain.from_iterable(scores_by_model[key]))
                     for key in scores_by_model}
    keys = sorted(global_scores.keys())
    data = [global_scores[key] for key in keys]
    produce_boxplot(data, keys, 'global')

    # for model_name in scores_by_model.keys():
    #     fig = plt.figure(figsize=(3, 7))
    #
    #     scores = scores_by_model[model_name]
    #
    #     medianprops = dict(linestyle='-.', linewidth=2.5, color='orange')
    #     plt.boxplot(scores,
    #                 patch_artist=True,
    #                 medianprops=medianprops)
    #     xtick_locs = [elem + 1 for elem in range(len(scores))]
    #     keys = ['Track {}'.format(i + 1) for i in range(len(scores))]
    #     plt.xticks(xtick_locs, keys)
    #     plt.xticks(rotation=90)
    #
    #     if model_name == 'mix':
    #         model_name = 'CNN'
    #     plt.title(model_name)
    #     fig.savefig('./test_figures/{}.png'.format(model_name))
    #     plt.close(fig)
        # plt.show()

    # for song_name in song_scores:
    #     data = [song_scores[song_name][key] for key in keys]
    #     produce_boxplot(data, keys, song_name)
