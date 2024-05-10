# This script uses the MTCFeatures package
# pip install mtcfeatures
# to get data and bare-minimum metadata on the Essen corpus
# and writes it out.

# let's run this from the FoNN directory. then dirname can be 'essen_corpus'

import os
import pandas as pd
from MTCFeatures import MTCFeatureLoader
import tqdm

features = {
    'midi_note_num': 'midipitch',
    'diatonic_note_num': 'diatonicpitch', 
    'chromatic_pitch_class': None,
    'beat_strength': 'beatstrength',
    'bar_num': None,
    'offset': None,
    'duration': 'duration',
    'velocity': None,
    'relative_chromatic_pitch': None,
    'relative_diatonic_pitch': None,
    'chromatic_scale_degree': None,
    'diatonic_scale_degree': 'scaledegree',
    'chromatic_interval': None,
    'diatonic_interval': None,
    'parsons_code': None,
    'parsons_cumsum': None
}

def mtcfeatures_metadata_to_csv():
    # writes metadata on all tunes to a single csv
    xs = list(fl.selectFeatures([]))
    rows = []
    for x in xs:
        row = {'identifiers': x['id'], 'origin': x['origin']}
        rows.append(row)
    df = pd.DataFrame(rows, columns=['identifiers','origin', 'title'])
    df.to_csv('essen_id_origin.csv')

def mtcfeatures_x_to_fonn_csv(x, dirname):
    # writes out a single csv storing feature data for a single tune
    df = pd.DataFrame(columns=list(features))
    for f in features:
        if features[f] is not None:
            data = x['features'][features[f]]
            df[f] = data
    df = df.fillna(0)
    outfilename = os.path.join(dirname, x['id']) + '.csv'
    print(outfilename)
    df.to_csv(outfilename)

def write_feature_sequence_all():
    # writes all the feature sequence csvs in a loop
    mtcfeatures_to_use = [features[x] for x in features if features[x] is not None]
    fl = MTCFeatureLoader('ESSEN')
    xs = fl.selectFeatures(mtcfeatures_to_use)

    for x in tqdm.tqdm(xs):
        mtcfeatures_x_to_fonn_csv(x, 'essen_corpus')


if __name__ == '__main__':
    write_feature_sequence_all()
    mtcfeatures_metadata_to_csv()