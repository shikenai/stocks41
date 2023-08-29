import pandas as pd
import time
import os
import sub
import matplotlib.pyplot as plt
import seaborn as sns

filename = '2topix_20230815.csv'
foloder_path = os.path.join(os.getcwd(), 'data', 'base')
full_name = os.path.join(foloder_path, filename)
t1 = time.time()
df = pd.read_csv(full_name, header=[0, 1])
df = df.swaplevel(axis=1).sort_index(axis=1)
df = df.reset_index(drop=True).drop(0)
df_date = df[["Symbols"]].droplevel(0, axis=1).rename(columns={"Attributes": "Date"})
df = df.drop("Symbols", axis=1)

_df_brands = pd.read_csv(os.path.join(foloder_path, "topix_all_20230814.csv"))
pd.set_option('display.max_rows', df.shape[0])
pd.set_option('display.max_columns', df.shape[1])

_brands_list = list(_df_brands["コード"])
brands_list = [str(b) + ".jp" for b in _brands_list]
# brands_list = brands_list[:1]
#
# ma_list = [3, 5, 7, 10, 12, 14, 15, 17, 20, 25, 30, 35, 40, 45, 50, 55, 60, 100, 120, 150, 180, 200]

# days_list = [3, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200]
days_list = [3, 5, 10, 15, 25, 60, 80, 100, 200]
# days_list = [3, 5]
# print(df.columns)
# print(brands_list)
# brands_list = brands_list[:1]
output_dir = os.path.join(os.getcwd(), "data", "output")
files = [s[:4] + ".jp" for s in os.listdir(output_dir)]

for brand in range(len(brands_list)):
    if brands_list[brand] in files:
        print(brands_list[brand], " is Done")
    else:
        print(brands_list[brand], ":go")
        _df = df[[brands_list[brand]]]
        _df.columns = _df.columns.droplevel(0)

        # 移動平均を追加する
        _df = sub.add_ma(_df, 'Close', days_list)

        # 追加した移動平均を基に、ゴールデン（デッド）クロスの発生、継続日数、勢いの大きさを追加する
        _df = sub.compare_ma(_df)
        # 追加した移動平均を基に、その移動平均が何日間継続しているかを追加する。
        # df = sub.set_continuous(df)
        _df = _df.drop(['Volume'], axis=1)

        target_days = 3  # 何日以内の収益を見込むか
        target_profit = 3  # 何％の収益を見込むか
        # 何日以内に何％の収益を見込むかを計算する。
        # 具体的には、全ての日の取引データについて、以下の処理を実行する。
        # ある日の翌日を１日目として、target_days日間以内に記録した最高値（最安値）を取得する。
        # （参考までに）前日がどの程度の陽線（陰線）なのかを取得する。
        # target_profitで指定した値以上に上昇したか（下落したか）を取得する。
        # 翌日の予想最高値（最安値）を取得する。
        _df = sub.set_predicted_rate(_df, days=target_days, target_plus=100 + target_profit,
                                     target_minus=100 - target_profit)

        sub.evaluate(_df, brands_list[brand])
        print(brands_list[brand], " is Done")

