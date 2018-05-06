import argparse
import itertools
import pathlib
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm

warnings.filterwarnings('ignore')


def read_data(data_file):
    df = pd.read_csv(data_file)

    df.date = pd.to_datetime(df.date)
    df = df.set_index('date')
    date_index = pd.date_range('2017-10-01', end='2018-04-30', freq='D')
    df = df.reindex(date_index)

    df.pv = df.pv.astype(np.float64)
    df['log_pv'] = np.log(df.pv)

    df_train = df[df.index < '2018-04-01']
    df_test = df[df.index >= '2018-04-01']
    return df_train, df_test


def get_score(ts_train, ts_test, params):
    model = SARIMAX(
        ts_train,
        order=(params['p'], params['d'], params['q']),
        seasonal_order=(params['sp'], params['sd'], params['sq'], 7),
        enforce_stationarity=False,
        enforce_invertibility=False)
    result = model.fit(disp=-1)
    train_pred = result.predict()
    test_pred = result.forecast(len(ts_test))
    score = {
        'aic': result.aic,
        'bic': result.bic,
        'train-rmse': np.sqrt(mean_squared_error(ts_train, train_pred)),
        'test-rmse': np.sqrt(mean_squared_error(ts_test, test_pred))
    }
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', type=pathlib.Path, default='data/processed/data.csv')
    parser.add_argument(
        '--out', type=pathlib.Path, default='data/processed/params.csv')
    parser.add_argument('--p_min', type=int, default=0)
    parser.add_argument('--p_max', type=int, default=4)
    parser.add_argument('--d_min', type=int, default=0)
    parser.add_argument('--d_max', type=int, default=1)
    parser.add_argument('--q_min', type=int, default=0)
    parser.add_argument('--q_max', type=int, default=2)
    parser.add_argument('--sp_min', type=int, default=0)
    parser.add_argument('--sp_max', type=int, default=1)
    parser.add_argument('--sd_min', type=int, default=0)
    parser.add_argument('--sd_max', type=int, default=1)
    parser.add_argument('--sq_min', type=int, default=0)
    parser.add_argument('--sq_max', type=int, default=1)
    args = parser.parse_args()

    df_train, df_test = read_data(args.input)
    ts_train = df_train.pv
    ts_test = df_test.pv

    p_range = range(args.p_min, args.p_max + 1)
    d_range = range(args.d_min, args.d_max + 1)
    q_range = range(args.q_min, args.q_max + 1)
    sp_range = range(args.sp_min, args.sp_max + 1)
    sd_range = range(args.sd_min, args.sd_max + 1)
    sq_range = range(args.sq_min, args.sq_max + 1)
    n = len(list(p_range)) * len(list(d_range)) * len(list(q_range)) * \
        len(list(sp_range)) * len(list(sd_range)) * len(list(sq_range))
    history = pd.DataFrame()
    for p, d, q, sp, sd, sq in tqdm(
            itertools.product(p_range, d_range, q_range, sp_range, sd_range,
                              sq_range),
            total=n):
        if p == 0 and q == 0:
            continue
        params = {'p': p, 'd': d, 'q': q, 'sp': sp, 'sd': sd, 'sq': sq}
        score = get_score(ts_train, ts_test, params)
        history = history.append(
            pd.Series({
                **params,
                **score
            }), ignore_index=True)
    history.sort_values(by='aic').to_csv(args.out)
    print(history.sort_values(by='aic').head())


if __name__ == '__main__':
    main()
