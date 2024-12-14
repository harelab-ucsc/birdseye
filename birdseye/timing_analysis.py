from dbConnector import dbConnector
import os
import numpy as np
import matplotlib.pyplot as plt


class testObj():
    def __init__(self, **kwargs):
        self.src_dir = kwargs.pop('src_dir', None)
        self.db_name = kwargs.pop('db_name', None)
        self.dbc = dbConnector(os.path.join(self.src_dir, self.db_name))


sensors = ['cam0', 'ahrs', 'rtk']
def extract_timestamps(sensor, db_name, dbc):
    try:
        return dbc.getFrom('time1, time2', f'{sensor}_images_{db_name}')
    except:
        return dbc.getFrom('time1, time2', f'{sensor}_data_{db_name}')


def between_timestamps(ts1, ts2):
    ret = []
    for item in zip(ts1, ts2):
        ret.append(item[0] - item[1])
    return ret


def within_timestamps(ts):
    ret = []
    for i in range(1, len(ts)):
        ret.append(ts[i] - ts[i-1])
    return ret


if __name__=='__main__':
    obj = testObj(src_dir='/home/mwmaster/parsed_flight', db_name='flight_data')
    resDict = {}
    print('extracting timestamps...')
    for sensor in sensors:
        resDict[sensor] = extract_timestamps(sensor, 'flight_data', obj.dbc)
        print(f'    {type(resDict[sensor])} {len(resDict[sensor])}, {type(resDict[sensor][0])} {len(resDict[sensor][0])}')
    print(' ', list(resDict.keys()))

    for key in resDict.keys():
        print(key)
        fig, ax = plt.subplots()
        tmp = np.array(resDict[key])
        ret = between_timestamps(tmp[:,0].tolist(), tmp[:,1].tolist())
        ret = np.array(ret)
        print('    between time1 and time2:', ret.mean(), ret.std())
        ax.hist(ret - ret.mean(), bins=50)
        ax.set_title(key)
        plt.show()
        del fig, ax

        fig, ax = plt.subplots(2)
        ret1 = within_timestamps(tmp[:,0].tolist())
        ret1 = np.array(ret1)
        ret2 = within_timestamps(tmp[:,1].tolist())
        ret2 = np.array(ret2)
        print('    within time1:', ret1.mean(), ret1.std())
        print('    within time2:', ret2.mean(), ret2.std())
        ax[0].hist(ret1 - ret1.mean(), bins=50)
        ax[1].hist(ret2 - ret2.mean(), bins=50)
        ax[0].set_title(key)
        plt.show()
        del fig, ax
