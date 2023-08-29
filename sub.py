import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import itertools
from itertools import product


def evaluate3(df):
    # ゴールデンクロスしている日の予想収益を取得
    print(df.columns)
    target_columns = [c for c in df.columns if c.endswith("change")]
    # print(target_columns)
    for target_column in target_columns:
        df_plus = df[df[target_column] == 1]
        df_minus = df[df[target_column] == -1]
        # print(_df.head(10))
        print(df["予想上昇率"].mean() - df_plus["翌日の最安値"].mean())
        print(df_plus["予想上昇率"].mean() - df_plus["翌日の最安値"].mean())
        print(df["予想下落率"].mean() - df_plus["翌日の最高値"].mean())
        print(df_minus["予想下落率"].mean() - df_plus["翌日の最高値"].mean())


def evaluate2(df):
    target_columns = [c for c in df.columns if c.endswith('positive')]
    target_plus = [c for c in df.columns if 'over' in c]
    target_minus = [c for c in df.columns if 'under' in c]
    df_all = df
    df_plus = df[df[target_plus[0]] == 1]
    df_minus = df[df[target_minus[0]] == 1]
    for col in target_columns:
        print(col)
        print('all')
        print(df_all[col].sum() / df_all.shape[0])
        print(df_all.head(10))
        print('plus')
        print(df_plus[df_plus[col] == 1][col].sum() / df_plus.shape[0])
        print(df_plus.head(10))
        print('minus')
        print(df_minus[df_minus[col] == -1][col].sum() / df_minus.shape[0])
        print(df_minus.head(10))


def evaluate(df, brand):
    change_columns = [c for c in df.columns if c.endswith('change')]
    over_column = [c for c in df.columns if 'over' in c]
    under_column = [c for c in df.columns if 'under' in c]
    # 各移動平均が上昇トレンドにあるか、下落トレンドにあるかを取得するため、positive列をリストで取得
    target_columns = [c for c in df.columns if c.endswith('positive')]
    entry_columns = [c for c in df.columns if '翌日' in c]
    profit_columns = [c for c in df.columns if '予想' in c]
    trigger = ["トリガー"]
    output_columns = target_columns + profit_columns + entry_columns + trigger
    output_df = pd.DataFrame(columns=output_columns)

    positive_columns = [[]]
    for n in range(1, len(target_columns) + 1):
        # 各移動平均positive列の数に応じて、その組み合わせを取得する。
        # この組み合わせは、あとで「上昇中」を判断するのに使用する。
        for c in itertools.combinations(target_columns, n):
            positive_columns.append(list(c))
    # 各移動平均のpositiveの組み合わせと併せて、これと相反する、negativeとなっている列を取得するため、組み合わせを取得する。
    negative_columns = []
    for positive_column in positive_columns:
        set_positive_columns = set(target_columns)
        set_temp_positive_columns = set(positive_column)
        negative_columns.append(list(set_positive_columns.symmetric_difference(set_temp_positive_columns)))

    for positive_column, negative_column in zip(positive_columns, negative_columns):
        _df = df
        # print("posi:")
        # print(positive_column)
        # print("nega:")
        # print(negative_column)
        for posi_col in positive_column:
            if not posi_col:
                # _df = df
                pass
            else:
                _df = _df[_df[posi_col] == 1]
        for nega_col in negative_column:
            _df = _df[_df[nega_col] == -1]

        for change_column in change_columns:
            temp_df_plus = _df[_df[change_column] == 1].reset_index(drop=True)
            temp_df_plus["トリガー"] = "+ " + change_column
            temp_df_minus = _df[_df[change_column] == -1].reset_index(drop=True)
            temp_df_minus["トリガー"] = "△ " + change_column
            mean_plus = temp_df_plus["予想上昇率"].mean()
            mean_low = temp_df_plus["翌日の最安値"].mean()
            mean_minus = temp_df_minus["予想下落率"].mean()
            mean_high = temp_df_minus["翌日の最高値"].mean()
            # print(temp_df_plus.head(5))
            # print(temp_df_minus.head(5))
            if not temp_df_plus.empty:
                # print(temp_df_plus.loc[0, output_columns])
                target_row = output_df.shape[0]
                output_df.loc[target_row, output_columns] = temp_df_plus.loc[0, output_columns]
                output_df.loc[target_row, "予想上昇率"] = mean_plus
                output_df.loc[target_row, "翌日の最安値"] = mean_low
                output_df.loc[target_row, "予想下落率"] = np.nan
                output_df.loc[target_row, "翌日の最高値"] = np.nan
            if not temp_df_minus.empty:
                # print(temp_df_minus.loc[0, output_columns])
                target_row = output_df.shape[0]
                output_df.loc[target_row, output_columns] = temp_df_minus.loc[0, output_columns]
                output_df.loc[target_row, "予想上昇率"] = np.nan
                output_df.loc[target_row, "翌日の最安値"] = np.nan
                output_df.loc[target_row, "予想下落率"] = mean_minus
                output_df.loc[target_row, "翌日の最高値"] = mean_high
    name = brand[:4] + ".csv"
    output_df.to_csv(os.path.join(os.getcwd(), "data", "output", name), encoding='shift-jis')


def add_ma(df, col, days_list):
    for days in days_list:
        df[f'{col}_{str(days)}ma'] = round(df[col].rolling(days).mean(), 1)
        df[f'{col}_{str(days)}ma_diff'] = df[f'{col}_{str(days)}ma'].diff()
        # df[f'{col}_{str(i)}ma_diff_rate'] = round(df[f'{col}_{str(i)}ma_diff'] / df[col] * 100, 3)
        df[f'{col}_{str(days)}ma_positive'] = df[f'{col}_{str(days)}ma_diff'].apply(lambda x: 1 if x > 0 else -1)
        df = df.drop(f'{col}_{str(days)}ma_diff', axis=1)
        df = df.dropna()

    return df


def set_continuous(df):
    columns_list = [c for c in list(df.columns) if c.endswith('positive') or ">" in c]
    # columns_list = [c for c in list(df.columns) if c.endswith('positive')]

    values = []

    df = df.reset_index()
    for col in columns_list:
        for n in range(df.shape[0]):
            if n == 0:
                pre = df.loc[n, col]
                values.append(pre)
            else:
                if df.loc[n, col] == df.loc[n - 1, col] == np.nan:
                    values.append(df.loc[n, col])
                elif df.loc[n, col] != df.loc[n - 1, col]:
                    pre = df.loc[n, col]
                    values.append(pre)
                elif df.loc[n, col] == df.loc[n - 1, col]:
                    pre = pre + df.loc[n, col]
                    values.append(pre)
        df[col + '_continuous'] = values
        values = []
    # print(df[['Close_3ma', 'Close_3ma_positive', 'Close_3ma_positive_continuous', 'Close_5ma', 'Close_5ma_positive',
    #           'Close_5ma_positive_continuous']])

    return df


def compare_ma(df):
    # maで終わる列名（移動平均）をリストにする
    _columns_list = [c for c in list(df.columns) if c.endswith('ma')]
    # 移動平均列名リストから、２つを比較して、各々取り出す
    columns_list = list(itertools.combinations(_columns_list, 2))
    for item in columns_list:
        idx = item[0].find('_')
        short = item[0][idx + 1:]
        long = item[1][idx + 1:]
        new_column_name = short + ">" + long
        # 短い移動平均が、長い平均を上回ったとき（ゴールデンクロス）に１を、下回ったとき（デッドクロス）に－1を入れる
        df[new_column_name] = np.where(df[item[0]] > df[item[1]], 1, -1)
        # ゴールデンクロス（orデッドクロス）が何日間継続しているかを入れる
        previous_value = np.nan
        values = []
        for i in df[new_column_name]:
            if pd.isnull(previous_value) or i != previous_value:
                if i == 1:
                    values.append(1)
                elif i == -1:
                    values.append(-1)
            else:
                values.append(0)
            previous_value = i
        # # 短期移動平均と長期移動平均の差分を取り、それを終値で割る（数字の絶対値が大きければ、直近の上昇率（下落率）が大きい）
        # df[f'{item[0].replace("Close_", "")}-{item[1].replace("Close_", "")}:momentum'] = round(
        #     (df[item[0]] - df[item[1]]) / df["Close"], 3) * 100
        # あとでごにょごにょするので、このタイミングでリネーム
        df[new_column_name.replace(">", " ") + ":change"] = values

    return df


def set_predicted_rate(df, days, target_plus, target_minus):
    max_list = []
    min_list = []
    index_num = df.shape[0]
    col_num = df.columns.get_loc('Close')
    for i in range(index_num):
        if i + days <= index_num:
            max_list.append(df.iloc[i + 1: i + days + 1, [col_num]].max().max())
            min_list.append(df.iloc[i + 1: i + days + 1, [col_num]].min().max())
        else:
            max_list.append(np.nan)
            min_list.append(np.nan)
    column_name_up = f'{days}日後までの最高値'
    column_name_down = f'{days}日後までの最安値'
    df[column_name_up] = max_list
    df[column_name_down] = min_list
    # df['（参考）前日のローソク足実体'] = (round((df['Close'] - df['Open']) / df['Close'], 3) * 100).shift(1)
    df["予想上昇率"] = round(df[column_name_up] / df["Close"], 3) * 100
    # df = judge_standard_number_plus(df, "予想上昇率", target_plus)
    df['shifted_Low'] = df["Low"].shift(-1)
    df['翌日の最安値'] = round(df['shifted_Low'] / df['Close'], 3) * 100
    df["予想下落率"] = round(df[column_name_down] / df["Close"], 3) * 100
    # df = judge_standard_number_minus(df, "予想下落率", target_minus)
    df['shifted_High'] = df["High"].shift(-1)
    df['翌日の最高値'] = round(df['shifted_High'] / df['Close'], 3) * 100
    df = df.drop([column_name_up, column_name_down, 'shifted_Low', 'shifted_High'], axis=1)
    df = df.dropna()

    return df


def judge_standard_number_plus(df, column, target):
    df[f'over_{str(target)}'] = (df[column] > target).astype(int)
    return df


def judge_standard_number_minus(df, column, target):
    df[f'under_{str(target)}'] = (df[column] < target).astype(int)
    return df


def make_img(df, columnA, columnB):
    # 列Bが1の場合の統計情報と可視化
    df_positive = df[df[columnB] == 1]
    mean_positive = df_positive[columnA].mean()
    median_positive = df_positive[columnA].median()
    sns.histplot(df_positive[columnA], kde=True, label=f'{columnB}=1')
    plt.axvline(mean_positive, color='r', linestyle='dashed', linewidth=2, label='Mean')
    plt.axvline(median_positive, color='g', linestyle='dashed', linewidth=2, label='Median')
    plt.text(mean_positive, 10, f'Mean_p: {mean_positive:.2f}', color='b', fontsize=10, ha='left')
    plt.legend()
    plt.title(f'Distribution of {columnA} when {columnB}=1')
    plt.xlabel(columnA)
    plt.ylabel('Frequency')
    plt.savefig('distribution_positive.png')  # グラフを画像として保存

    # 列Bが-1の場合の統計情報と可視化
    df_negative = df[df[columnB] == -1]
    mean_negative = df_negative[columnA].mean()
    median_negative = df_negative[columnA].median()
    sns.histplot(df_negative[columnA], kde=True, label=f'{columnB}=-1')
    plt.axvline(mean_negative, color='r', linestyle='dashed', linewidth=2, label='Mean')
    plt.axvline(median_negative, color='g', linestyle='dashed', linewidth=2, label='Median')
    plt.text(mean_negative, 10, f'Mean_n: {mean_negative:.2f}', color='b', fontsize=10, ha='right')
    plt.legend()
    plt.title(f'Distribution of {columnA} when {columnB}=-1')
    plt.xlabel(columnA)
    plt.ylabel('Frequency')
    plt.savefig('distribution_negative.png')  # グラフを画像として保存

    # # 結果のCSVファイル保存
    df_positive.to_csv('data_positive.csv', index=False)
    df_negative.to_csv('data_negative.csv', index=False)

    plt.show()  # グラフを表示
