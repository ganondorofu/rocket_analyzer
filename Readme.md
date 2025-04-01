# 針路可視化できるマン

csvデータを元にPythonのフレームワークであるplotlyを用いて飛行針路を描画するコードです

## つかいかた
data.pyと同じディレクトリに処理したいcsvファイルを入れてください。

22行目のファイル名を確認してください。異なる場合は、以下のコード内のファイル名を変更してください。

```python
df = pd.read_csv("sensor_data50.csv")
```

## 必要なライブラリのインストールコマンド

以下のコマンドをシェルに入力して、必要なライブラリをインストールしてください。

Pythonの仮想環境を作成してからインストールすることを推奨します。

```shell
pip install pandas
pip install scipy
pip install plotly
```

## csvファイル内カラムの説明
### Time_s

見ての通り時間です

- 3/31の実験でのデータでは飛行判定前と後のファイルに分かれていてそのファイル間で1秒ほど記録開始にラグがあります

Temperature_C,Humidity_%,Pressure_hPa,Lat,Lng,Alt_m,Fix,Sats,AccelX_g,AccelY_g,AccelZ_g,GyroX_deg_s,GyroY_deg_s,GyroZ_deg_s,TotalAccel