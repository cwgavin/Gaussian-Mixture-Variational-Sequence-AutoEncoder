import csv
import json
from collections import defaultdict
from datetime import datetime
import time
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import pickle

# Longitude and latitude coordinates of Porto
# (x, y) = (lon, lat)
# (150, 50)
# lon_mid = -8.61099
# lat_mid = 41.14961

# lon_mid = -8.628740
# lat_mid = 41.158255

lon_mid = -8.598054
lat_mid = 41.161620
lon_offset = 7500 / 83940  # = 0.0893495    7500 meters  1m = 1/83940 deg
lat_offset = 2500 / 111060  # = 0.0225104
lon_min, lon_max = lon_mid - lon_offset, lon_mid + lon_offset
lat_min, lat_max = lat_mid - lat_offset, lat_mid + lat_offset


def get_grid_id(t, map_size):
    global lat_min, lat_max, lon_min, lon_max
    traj = []
    # grid_width = (2*lon_offset) / map_size[0]
    # grid_height = (2*lat_offset) / map_size[1]
    grid_width = 100 / 83940
    grid_height = 100 / 111060
    for [x, y] in t:
        grid_x = math.ceil((x - lon_min) / grid_width)
        grid_y = math.floor((lat_max - y) / grid_height)
        grid_id = grid_y * map_size[0] + grid_x
        traj.append(grid_id)
    return traj

def count_sd(x):
    global sd
    sd[(x[0], x[-1])] += 1

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def add_timestamp (row):
    traj = row['traj']
    for i in range(len(traj)):
        t = row['time'] + i * 15    # each 15 seconds
        # timeslot = (0 if row['TIMESTAMP'].weekday() < 5 else 1) * 24 + (t // 3600)
        timeslot = (0 if row['TIMESTAMP'].weekday() < 5 else 1) * 24 + (t // 3600 + 1)
        traj[i] = (traj[i], timeslot)
    return traj


def process(csv_name, skiprows, map_size, delimiter, quoting, output_file):
    # bins = 150 * 50  # = 7500
    # bins = [50, 150]
    bins = [2000, 2000]
    z = np.zeros(bins)
    total_rows = 0
    add_header = True
    write_mode = 'w'
    data = pd.read_csv(csv_name,
                       chunksize=10000,
                       usecols=['TIMESTAMP', 'POLYLINE'],
                       skiprows=skiprows,
                       iterator=True,
                       converters={'TIMESTAMP': lambda x: datetime.utcfromtimestamp(int(x)),
                                   'POLYLINE': lambda x: json.loads(x)})
    for chunk in data:
        # chunk = chunk[chunk['POLYLINE'].apply(lambda x: len(x) > 1
        chunk = chunk[chunk['POLYLINE'].apply(lambda x: len(x) > 2
                                                        and False not in [lon_max >= c[0] >= lon_min
                                                                          and lat_max >= c[1] >= lat_min
                                                                          for c in x]
                                              )]
        # chunk['weekday'] = chunk['TIMESTAMP'].apply(lambda x: 1 if x.weekday() < 5 else 0)
        chunk['time'] = chunk['TIMESTAMP'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
        chunk['timeslot'] = chunk['TIMESTAMP'].apply(lambda x: (0 if x.weekday() < 5 else 1) * 24 + x.hour)
        chunk['traj'] = chunk['POLYLINE'].apply(get_grid_id, args=(map_size,))
        chunk['traj'].apply(count_sd)

        chunk['traj'] = chunk.apply(add_timestamp, axis=1)
        # df['period'] = df[['Year', 'quarter']].apply(lambda x: ''.join(x), axis=1)
        # chunk['traj'] = chunk['POLYLINE'].apply(add_timestamp, args=(chunk['timeslot'],))

        latlon = np.array([(lat, lon)
                           for path in chunk.POLYLINE
                           for lon, lat in path if len(path) > 0])
        z += np.histogram2d(x=latlon.T[0], y=latlon.T[1], bins=bins, range=[[lat_min, lat_max],[lon_min, lon_max]])[0]

        chunk = chunk.drop(["TIMESTAMP", "POLYLINE"], axis=1)
        total_rows += chunk.shape[0]
        chunk.to_csv(path_or_buf=output_file, index=False, header=add_header, mode=write_mode, sep=delimiter, quoting=quoting)
        if add_header:
            add_header = False
            write_mode = 'a'
    print(f'Total rows: {total_rows} (trajectories within the selected area)')
    return z

def draw_fig(z, img_file):
    # global z, lon_min, lon_max, lat_min, lat_max, total_rows
    log_density = np.log(1+z)
    plt.imshow(log_density[::-1,:], # flip vertically
               extent=[lon_min, lon_max, lat_min, lat_max])
    # ax = plt.subplot(1,1,1)
    # plt.imshow(z[::-1,:], extent=[lon_min, lon_max, lat_min, lat_max])
    # cbar = plt.colorbar()
    # cbar.ax.set_ylabel('Counts')
    plt.savefig(img_file)

def filter_by_sd(delimiter, quoting, output_file, output_file2):
    sd = load_obj('sd')
    data = pd.read_csv(output_file,
                       chunksize=10000,
                       iterator = True,
                       delimiter=delimiter,
                       # usecols=['timeslot', 'traj'],
                       usecols=['traj'],
                       # converters={'traj': lambda x: json.loads(x)})
                       converters = {'traj': lambda x: eval(x)})
    total_rows = 0
    write_mode = 'w'
    for chunk in data:
        chunk = chunk[chunk['traj'].apply(lambda x: sd[(x[0][0], x[-1][0])] >= 25)]
        total_rows += chunk.shape[0]

        train_set = chunk.sample(frac=0.8, random_state=0)
        test_set = chunk.drop(train_set.index)

        train_set.to_csv(path_or_buf=output_file2+'_train.csv', index=False, header=False, mode=write_mode, sep=delimiter, quoting=quoting)
        test_set.to_csv(path_or_buf=output_file2+'_val.csv', index=False, header=False, mode=write_mode, sep=delimiter,quoting=quoting)
        if write_mode == 'w':
            write_mode = 'a'
    print(f'Total rows: {total_rows} (trajectories after sd filtering)')


map_size = [150, 50]
skiprows = None
# skiprows = lambda x: x > 20000
start = time.time()
sd = defaultdict(int)

delimiter = '_'
quoting = csv.QUOTE_NONE
input_csv = 'porto_train_raw.csv'
output_file = f'data/p_{datetime.now().date()}_0.csv'
output_file2 = f'data/p_{datetime.now().date()}_1'
img_file = 'img/script1.png'

z = process(input_csv, skiprows, map_size, delimiter, quoting, output_file)
save_obj(sd, 'sd')
# draw_fig(z, img_file)
filter_by_sd(delimiter, quoting, output_file, output_file2)

end = time.time()
print(f'Elapsed time: {int(end - start)} seconds = {round(int(end - start) / 60, 1)} minutes')
