from collections import defaultdict
import pandas as pd
# import dask.dataframe as dd
import datetime
import time
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# def convert_timestamp(t):
#     utc_datetime = datetime.datetime.fromtimestamp(t)
#     tz_datetime = utc_datetime.astimezone(timezone('Portugal'))
#     return tz_datetime

def convert_timestamp(t):
    return datetime.datetime.utcfromtimestamp(t)

# def get_extreme_coordinates(t):
#     global top_left, top_right, bottom_left, bottom_right
#     for coordinate in eval(t):
#         if coordinate[0] < top_left[0] and coordinate[1] > top_left[1]:
#             top_left = coordinate
#         elif coordinate[0] > top_right[0] and coordinate[1] > top_right[1]:
#             top_right = coordinate
#         elif coordinate[0] < bottom_left[0] and coordinate[1] < bottom_left[1]:
#             bottom_left = coordinate
#         elif coordinate[0] > bottom_right[0] and coordinate[1] < bottom_right[1]:
#             bottom_right = coordinate

def get_extreme_values(t):
    global x_left, x_right, y_top, y_bottom, gx, gy
    for [x, y] in eval(t):
        gx.append(x)
        gy.append(y)
        # if abs(x - x_left) > 0.5 or abs(x - x_right) > 0.5:
        #     pass
        if x < x_left:
            x_left = x
        elif x > x_right:
            x_right = x
        # if abs(y - y_top) > 0.5 or abs(y - y_bottom) > 0.5:
        #     pass
        if y > y_top:
            y_top = y
        elif y < y_bottom:
            y_bottom = y

def get_grid_id(t, map_size):
    global x_left, x_right, y_top, y_bottom
    traj = []
    grid_width = (x_right - x_left) / map_size[0]
    grid_height = (y_top - y_bottom) / map_size[1]
    for [x, y] in eval(t):
        grid_x = math.ceil((x - x_left) / grid_width)
        grid_y = math.floor((y_top - y) / grid_height)
        grid_id = grid_y * map_size[0] + grid_x
        traj.append(grid_id)
    return traj

def count_sd(x):
    global sd
    sd[(x[0], x[-1])] += 1

def show_plot(df, map_size):
    global gx, gy
    mpl.rcParams['agg.path.chunksize'] = 10000
    # for p in traj:
    #     if not p == 0:
    #         gy.append(int(p // map_size[0]))
    #         gx.append(int(p % map_size[0]))
    plt.plot(gx, gy, alpha=0.6)
    plt.show()

MAP_SIZE = [100, 100]
start = time.time()
# df = pd.read_csv('porto_test_raw.csv')
dtype = {'TIMESTAMP': 'int', 'POLYLINE': 'str'}
df = pd.read_csv('porto_train_raw.csv', usecols=['TIMESTAMP', 'POLYLINE'], dtype=dtype)
# df = pd.read_csv('porto_train_raw.csv', usecols=['TIMESTAMP', 'POLYLINE'], dtype=dtype, skiprows = lambda x: x > 20000)
# df = dd.read_csv('porto_train_raw.csv', usecols=['TIMESTAMP', 'POLYLINE'], dtype=dtype)
# df = df[df['POLYLINE'].map(lambda x: len(eval(x)) >= 1)]
# df = df[df['POLYLINE'] != '[]']
df = df[df['POLYLINE'].apply(lambda x: len(eval(x)) > 1)]

df['datetime'] = df['TIMESTAMP'].apply(convert_timestamp)
df['weekday'] = df['datetime'].apply(lambda x: 1 if x.weekday() < 5 else 0)
df['time'] = df['datetime'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)
# df['datetime'] = df['TIMESTAMP'].apply(convert_timestamp, meta=('TIMESTAMP', 'datetime64[ns]'))
# df['weekday'] = df['datetime'].apply(lambda x: x.weekday() < 5, meta=('datetime', 'bool'))
# df['time'] = df['datetime'].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second, meta=('datetime', 'int64'))

# top_left = top_right = bottom_left = bottom_right = eval(df['POLYLINE'][0])[0]
first_coordinate = eval(df['POLYLINE'][0])[0]
# first_coordinate = eval(df.compute()['POLYLINE'][0])[0]
x_left = x_right = first_coordinate[0]
y_top = y_bottom = first_coordinate[1]

gx = []
gy = []
df['POLYLINE'].apply(get_extreme_values)

df['traj'] = df['POLYLINE'].apply(get_grid_id, args=(MAP_SIZE,))
print(x_left, x_right, y_top, y_bottom)
print(f'Total rows: {len(df.index)}')

show_plot(df, MAP_SIZE)

sd = defaultdict(int)
df['traj'].apply(count_sd)
df = df[df['traj'].apply(lambda x: sd[(x[0], x[-1])] >= 25)]


# # df = df.drop(["TRIP_ID", "CALL_TYPE", "ORIGIN_CALL", "ORIGIN_STAND", "TAXI_ID", "TIMESTAMP", "DAY_TYPE", "MISSING_DATA", "POLYLINE"], axis=1)
df = df.drop(["TIMESTAMP", "POLYLINE", "datetime"], axis=1)
df.to_csv(path_or_buf='processed.csv', index=False, header=False)
end = time.time()
print(f'elapsed time: {int(end - start)} seconds = {round(int(end - start) / 60, 1)} minutes')
