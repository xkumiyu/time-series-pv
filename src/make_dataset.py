import argparse
import pathlib

import numpy as np
import pandas as pd


def read_pv(file):
    df = pd.read_csv(file, header=5).dropna()
    df = df.rename(columns={'日の指標': 'date', 'ページビュー数': 'pv'})
    df.date = pd.to_datetime(df.date)
    df.pv = df.pv.apply(lambda x: int(x.replace(',', '')))
    df = df.set_index('date')
    return df


def read_entry(file):
    df = pd.read_csv(file, header=None)
    df = df.rename(columns={0: 'date'})
    df.date = pd.to_datetime(df.date.str[6:16])
    df = df.sort_values('date')
    df['entry'] = range(1, len(df) + 1)
    df = df.set_index('date')
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ga',
        dest='google_analytics_file',
        type=pathlib.Path,
        default='data/raw/Analytics すべてのウェブサイトのデータ ユーザー サマリー 20171001-20180430.csv')
    parser.add_argument(
        '--ed',
        dest='entry_date_file',
        type=pathlib.Path,
        default='data/interim/entry_date.txt')
    parser.add_argument(
        '--out', type=pathlib.Path, default='data/processed/data.csv')
    args = parser.parse_args()

    if not args.google_analytics_file.exists(
    ) or not args.entry_date_file.exists():
        print('No files found.')
        return

    df = read_pv(args.google_analytics_file).join(
        read_entry(args.entry_date_file))
    df = df.sort_index()
    df.iloc[0, 1] = 8
    df.entry = df.entry.interpolate().apply(np.floor)
    df.entry = pd.to_numeric(df.entry, downcast='integer')
    df['pv_per_entry'] = df.pv / df.entry
    df = df[~df.index.duplicated(keep='first')]

    df.to_csv(args.out)


if __name__ == '__main__':
    main()
