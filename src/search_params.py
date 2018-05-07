import argparse
import itertools
import pathlib
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, tpe
from statsmodels.tsa.statespace.sarimax import SARIMAX

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


def grid_search(data, ranges):
    ts_train = data[0].pv
    ts_test = data[1].pv
    n = len(list(ranges['p'])) * len(list(ranges['d'])) * len(list(ranges['q'])) * \
        len(list(ranges['sp'])) * \
        len(list(ranges['sd'])) * len(list(ranges['sq']))
    history = pd.DataFrame()
    for p, d, q, sp, sd, sq in tqdm(
            itertools.product(ranges['p'], ranges['d'], ranges['q'],
                              ranges['sp'], ranges['sd'], ranges['sq']),
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
    return history


def hyperopt_search(data, ranges, max_evals=100):
    ts_train = data[0].pv
    ts_test = data[1].pv

    space = {
        'p': hp.randint('p', ranges['p'][-1] + 1),
        'd': hp.randint('d', ranges['d'][-1] + 1),
        'q': hp.randint('q', ranges['q'][-1] + 1),
        'sp': hp.randint('sp', ranges['sp'][-1] + 1),
        'sd': hp.randint('sd', ranges['sd'][-1] + 1),
        'sq': hp.randint('sq', ranges['sq'][-1] + 1),
    }

    # space = {
    #     'p': hp.quniform('p', ranges['p'][0], ranges['p'][-1], 1),
    #     'd': hp.quniform('d', ranges['d'][0], ranges['d'][-1], 1),
    #     'q': hp.quniform('q', ranges['q'][0], ranges['q'][-1], 1),
    #     'sp': hp.quniform('sp', ranges['sp'][0], ranges['sp'][-1], 1),
    #     'sd': hp.quniform('sd', ranges['sd'][0], ranges['sd'][-1], 1),
    #     'sq': hp.quniform('sq', ranges['sq'][0], ranges['sq'][-1], 1),
    # }
    # space = {
    #     'p': hp.choice('p', ranges['p']),
    #     'd': hp.choice('d', ranges['d']),
    #     'q': hp.choice('q', ranges['q']),
    #     'sp': hp.choice('sp', ranges['sp']),
    #     'sd': hp.choice('sd', ranges['sd']),
    #     'sq': hp.choice('sq', ranges['sq']),
    # }

    pbar = tqdm(total=max_evals)

    def objective(params):
        pbar.update(1)
        if params['p'] == 0 and params['q'] == 0:
            return {'status': STATUS_FAIL}
        else:
            score = get_score(ts_train, ts_test, params)
            return {'loss': score['aic'], 'status': STATUS_OK}

    trials = Trials()
    best = fmin(
        objective, space, algo=tpe.suggest, trials=trials, max_evals=max_evals)
    pbar.close()
    return best, trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', type=pathlib.Path, default='data/processed/data.csv')
    parser.add_argument(
        '--out', type=pathlib.Path, default='data/processed/params.csv')
    parser.add_argument(
        '-s',
        '--search_type',
        choices=['grid', 'hyperopt'],
        default='hyperopt')
    parser.add_argument('-n', '--max_evals', type=int, default=100)
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

    data = read_data(args.input)
    ranges = {
        'p': range(args.p_min, args.p_max + 1),
        'd': range(args.d_min, args.d_max + 1),
        'q': range(args.q_min, args.q_max + 1),
        'sp': range(args.sp_min, args.sp_max + 1),
        'sd': range(args.sd_min, args.sd_max + 1),
        'sq': range(args.sq_min, args.sq_max + 1)
    }

    if args.search_type == 'grid':
        history = grid_search(data, ranges)
        history.sort_values(by='aic').to_csv(args.out)
        print(history.sort_values(by='aic').head())
    elif args.search_type == 'hyperopt':
        best, trials = hyperopt_search(data, ranges, args.max_evals)
        print('best params = {}'.format(best))
        # print(trials.trials)


if __name__ == '__main__':
    main()
