from datetime import timedelta

import numpy as np
import pandas as pd
from tqdm import trange

from cr_utils import check_time


@check_time
def query_to_df(data: tuple, columns: str) -> pd.DataFrame:
    data = pd.DataFrame.from_records(data)
    data.columns = list(map(lambda x: x.split(' ')[-1], columns.split(',')))
    return data


@check_time
def rearrange_by_time(data) -> dict[str, pd.DataFrame]:
    parameter_id = set(data['PARAMETER_ID'].values.tolist())
    motor, wire = [], []
    for p in parameter_id:
        if '000_00001' in p:
            wire.append(p)
        elif '000_00002' in p:
            motor.append(p)

    motor.sort()
    wire.sort()
    motor = {
        m: data[data['PARAMETER_ID']==m].sort_values('TXN_TIME') for m in motor}
    wire = {
        w: data[data['PARAMETER_ID']==w].sort_values('TXN_TIME') for w in wire}
    motor_result = {'datetime': list(motor.values())[0]['TXN_TIME'].values}
    wire_result = {'datetime': list(wire.values())[0]['TXN_TIME'].values}
    motor = {k: v['VALUE'].values.astype(float) for k, v in motor.items()}
    wire = {k: v['VALUE'].values.astype(float) for k, v in wire.items()}
    motor_result.update(motor)
    wire_result.update(wire)

    return {
        'motor': pd.DataFrame(motor_result), 'wire': pd.DataFrame(wire_result)}


def calc_trend(data: np.ndarray|list|tuple, dim: int = 1) -> np.ndarray:
    num = len(data)
    x = np.arange(num)
    return np.polynomial.Polynomial.fit(x, data, dim).linspace(num)[1]


def calc_mb(data: np.ndarray|list|tuple, dim: int = 1) -> np.ndarray:
    x = np.arange(len(data))
    bm = np.polynomial.polynomial.polyfit(x, data, dim).T
    return bm[:, ::-1]  # bm -> mb


@check_time
def make_dataset(data: pd.DataFrame,
                 day: int = 3,
                 hour: int = 0,
                 window_day: int = 0,
                 window_hour: int = 1,
                 interval: int = 1,
) -> dict[str, np.ndarray]:
    num = 1e+10
    limit = (3600/interval) * (day*24+hour) * .99
    _timedelta = timedelta(days=day, hours=hour)
    _time_window = timedelta(days=window_day, hours=window_hour)
    key = data.columns[0]  # datetime
    columns = data.columns[1:]
    result = []
    from_dt = data[key].iloc[0]
    dt = data[key].iloc[-1] - from_dt
    total = np.ceil(np.ceil((dt.days*24*3600 + dt.seconds) / 3600)
                    / (window_day*24 + window_hour)).astype(int) + 1
    for _ in trange(total, desc='make_dataset'):
        to_dt = from_dt + _timedelta
        res = data[((from_dt<=data[key]) & (data[key]<to_dt))]
        if len(res) > limit:
            mb = calc_mb(res[columns])
            result.append(np.concatenate(
                [mb,
                 res[columns].mean().values[..., None],
                 res[columns].std().values[..., None]],
                axis=1))

        from_dt += _time_window

    result = np.reshape(np.concatenate(result, 1), (len(columns), -1, 4))
    result = {k: v for k, v in zip(columns, result)}
    return result


@check_time
def make_dataset_dt(data: pd.DataFrame,
                    day: int = 3,
                    hour: int = 0,
                    window_day: int = 0,
                    window_hour: int = 1,
                    interval: int = 1,
) -> dict[str, np.ndarray]:
    num = 1e+10
    limit = (3600/interval) * (day*24+hour) * .99
    _timedelta = timedelta(days=day, hours=hour)
    _time_window = timedelta(days=window_day, hours=window_hour)
    key = data.columns[0]  # datetime
    columns = data.columns[1:]
    result = []
    result_dt = []
    from_dt = data[key].iloc[0]
    dt = data[key].iloc[-1] - from_dt
    total = np.ceil(np.ceil((dt.days*24*3600 + dt.seconds) / 3600)
                    / (window_day*24 + window_hour)).astype(int) + 1
    for _ in trange(total, desc='make_dataset'):
        to_dt = from_dt + _timedelta
        res = data[((from_dt<=data[key]) & (data[key]<to_dt))]
        if len(res) >= limit:
            mb = calc_mb(res[columns])
            result.append(np.concatenate(
                [mb,
                 res[columns].mean().values[..., None],
                 res[columns].std().values[..., None]],
                axis=1))
            result_dt.append([res[key].iloc[0], res[key].iloc[-1]])

        from_dt += _time_window

    result = np.reshape(np.concatenate(result, 1), (len(columns), -1, 4))
    result = {k: v for k, v in zip(columns, result)}
    return {'datetime': pd.DataFrame(result_dt, columns=['start', 'finish']),
            'data': result}
