# coding: utf-8

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import cumulative_trapezoid
import math

# -------------------------------------------------------------
# 複数候補の列名から存在するものを選ぶヘルパー関数
# -------------------------------------------------------------
def select_column(df, alternatives):
    for alt in alternatives:
        if alt in df.columns:
            return alt
    raise ValueError(f"必要な列 {alternatives} がCSVに含まれていません。")

# -------------------------------------------------------------
# オイラー角（roll, pitch, yaw）から回転行列を計算するクラス
# -------------------------------------------------------------
class RotationMatrixCalculator:
    def __init__(self):
        pass
    
    def euler_to_rotation_matrix(self, roll, pitch, yaw):
        # X軸回りの回転行列
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        # Y軸回りの回転行列
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        # Z軸回りの回転行列
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        R = R_z @ R_y @ R_x
        return R

# -------------------------------------------------------------
# メイン処理
# -------------------------------------------------------------
def main():
    # CSVファイルのパス（例）
    csv_file = "sensor_data.csv"
    
    # CSVファイルからセンサーデータを読み込む
    # コメント行(#始まり)、空行をスキップし、不正行があってもスキップする
    df = pd.read_csv(
        csv_file,
        comment='#',
        skip_blank_lines=True,
        on_bad_lines='skip'
    )
    
    # 重複ヘッダー行（"Time_s"が文字列のままの行）を除外する
    if 'Time_s' in df.columns:
        df = df[df['Time_s'].astype(str) != 'Time_s']
    
    # 数値変換（変換できないセルはNaNとなる）
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Time_s 列が存在しない場合はエラー
    if 'Time_s' not in df.columns:
        raise ValueError("CSVに 'Time_s' 列が存在しません。")
    # Time_s 列にNaNが含まれる行を削除
    df = df.dropna(subset=['Time_s'])
    
    # タイムスタンプ配列の取得
    time = df['Time_s'].values
    
    # -------------------------------
    # 必要な列を候補名から選択
    # -------------------------------
    temperature_col = select_column(df, ["Temperature", "Temperature_C"])
    humidity_col    = select_column(df, ["Humidity", "Humidity_%"])
    pressure_col    = select_column(df, ["Pressure", "Pressure_hPa"])
    accelx_col      = select_column(df, ["AccelX_g"])
    accely_col      = select_column(df, ["AccelY_g"])
    accelz_col      = select_column(df, ["AccelZ_g"])
    gyrox_col       = select_column(df, ["GyroX_deg", "GyroX_deg_s"])
    gyroy_col       = select_column(df, ["GyroY_deg", "GyroY_deg_s"])
    gyroz_col       = select_column(df, ["GyroZ_deg", "GyroZ_deg_s"])
    
    # -------------------------------
    # 加速度（g → m/s^2）とジャイロ（deg/s → rad/s）の取得
    # -------------------------------
    # 加速度は g 単位なので 9.81 を掛ける
    acc_data = df[[accelx_col, accely_col, accelz_col]].to_numpy() * 9.81
    # ジャイロは deg/s → rad/s に変換
    gyro_data = df[[gyrox_col, gyroy_col, gyroz_col]].to_numpy()
    gyro_data_rad = np.deg2rad(gyro_data)
    
    # -------------------------------
    # ジャイロデータからオイラー角（roll, pitch, yaw）を積分で算出
    # -------------------------------
    attitude = cumulative_trapezoid(gyro_data_rad, x=time, initial=0, axis=0)
    
    # -------------------------------
    # ローカル加速度からグローバル加速度への変換
    # -------------------------------
    rot_calc = RotationMatrixCalculator()
    N = len(time)
    global_acc = np.zeros_like(acc_data)
    for i in range(N):
        roll = attitude[i, 0]
        pitch = attitude[i, 1]
        yaw = attitude[i, 2]
        R_mat = rot_calc.euler_to_rotation_matrix(roll, pitch, yaw)
        global_acc[i] = R_mat @ acc_data[i]
    
    # 重力 (9.81 m/s^2) をグローバルZ軸から除去
    global_acc[:, 2] -= 9.81
    
    # -------------------------------
    # 各時刻のグローバル加速度の大きさ（ノルム）を計算
    # -------------------------------
    acc_magnitude = np.linalg.norm(global_acc, axis=1)
    
    # -------------------------------
    # 台形公式で加速度→速度→位置を積分計算
    # -------------------------------
    velocity = cumulative_trapezoid(global_acc, x=time, initial=0, axis=0)
    position = cumulative_trapezoid(velocity, x=time, initial=0, axis=0)
    
    # -------------------------------
    # アニメーション用フレーム作成用のインデックス作成
    # データ行数が50未満の場合は全行、50以上の場合は最大100フレームにサンプリング
    # -------------------------------
    if N < 50:
        indices = list(range(N))
    else:
        max_frames = 100
        step = max(1, N // max_frames)
        indices = list(range(0, N, step))
        if indices[-1] != N - 1:
            indices.append(N - 1)
    
    frames = []
    axis_length = 1.0  # 局所座標軸の長さ（メートル）
    
    # 飛行軌跡（全時刻の位置）のトレースを作成
    flight_path_trace = go.Scatter3d(
        x=position[:, 0],
        y=position[:, 1],
        z=position[:, 2],
        mode='lines',
        line=dict(
            color=acc_magnitude,
            colorscale='Viridis',
            width=5,
            colorbar=dict(
                title=dict(text='加速度 (m/s^2)'),
                tickfont=dict(size=10)
            )
        ),
        name='飛行軌跡',
        customdata=acc_magnitude,
        hovertemplate='加速度: %{customdata:.2f} m/s^2<br>X: %{x:.2f} m<br>Y: %{y:.2f} m<br>Z: %{z:.2f} m'
    )
    
    # 各フレームごとに、現在位置と局所座標軸を表示するためのトレースを作成
    for idx in indices:
        pos_current = position[idx]
        roll = attitude[idx, 0]
        pitch = attitude[idx, 1]
        yaw = attitude[idx, 2]
        
        R_mat = rot_calc.euler_to_rotation_matrix(roll, pitch, yaw)
        x_axis_end = pos_current + R_mat @ np.array([axis_length, 0, 0])
        y_axis_end = pos_current + R_mat @ np.array([0, axis_length, 0])
        z_axis_end = pos_current + R_mat @ np.array([0, 0, axis_length])
        
        current_pos_trace = go.Scatter3d(
            x=[pos_current[0]],
            y=[pos_current[1]],
            z=[pos_current[2]],
            mode='markers',
            marker=dict(size=5, color='red'),
            name='現在位置'
        )
        x_axis_trace = go.Scatter3d(
            x=[pos_current[0], x_axis_end[0]],
            y=[pos_current[1], x_axis_end[1]],
            z=[pos_current[2], x_axis_end[2]],
            mode='lines',
            line=dict(color='red', width=5),
            name='X軸'
        )
        y_axis_trace = go.Scatter3d(
            x=[pos_current[0], y_axis_end[0]],
            y=[pos_current[1], y_axis_end[1]],
            z=[pos_current[2], y_axis_end[2]],
            mode='lines',
            line=dict(color='green', width=5),
            name='Y軸'
        )
        z_axis_trace = go.Scatter3d(
            x=[pos_current[0], z_axis_end[0]],
            y=[pos_current[1], z_axis_end[1]],
            z=[pos_current[2], z_axis_end[2]],
            mode='lines',
            line=dict(color='orange', width=5),
            name='Z軸'
        )
        
        frame_data = [
            flight_path_trace,
            current_pos_trace,
            x_axis_trace,
            y_axis_trace,
            z_axis_trace
        ]
        frames.append(go.Frame(data=frame_data, name=f"frame{idx}"))
    
    # 初期フレームの設定
    init_idx = indices[0]
    pos_init = position[init_idx]
    roll_init = attitude[init_idx, 0]
    pitch_init = attitude[init_idx, 1]
    yaw_init = attitude[init_idx, 2]
    R_init = rot_calc.euler_to_rotation_matrix(roll_init, pitch_init, yaw_init)
    
    x_axis_end_init = pos_init + R_init @ np.array([axis_length, 0, 0])
    y_axis_end_init = pos_init + R_init @ np.array([0, axis_length, 0])
    z_axis_end_init = pos_init + R_init @ np.array([0, 0, axis_length])
    
    current_pos_trace_init = go.Scatter3d(
        x=[pos_init[0]],
        y=[pos_init[1]],
        z=[pos_init[2]],
        mode='markers',
        marker=dict(size=5, color='red'),
        name='現在位置'
    )
    x_axis_trace_init = go.Scatter3d(
        x=[pos_init[0], x_axis_end_init[0]],
        y=[pos_init[1], x_axis_end_init[1]],
        z=[pos_init[2], x_axis_end_init[2]],
        mode='lines',
        line=dict(color='red', width=5),
        name='X軸'
    )
    y_axis_trace_init = go.Scatter3d(
        x=[pos_init[0], y_axis_end_init[0]],
        y=[pos_init[1], y_axis_end_init[1]],
        z=[pos_init[2], y_axis_end_init[2]],
        mode='lines',
        line=dict(color='green', width=5),
        name='Y軸'
    )
    z_axis_trace_init = go.Scatter3d(
        x=[pos_init[0], z_axis_end_init[0]],
        y=[pos_init[1], z_axis_end_init[1]],
        z=[pos_init[2], z_axis_end_init[2]],
        mode='lines',
        line=dict(color='orange', width=5),
        name='Z軸'
    )
    
    # Plotlyのレイアウトとアニメーション設定
    fig = go.Figure(
        data=[
            flight_path_trace,
            current_pos_trace_init,
            x_axis_trace_init,
            y_axis_trace_init,
            z_axis_trace_init
        ],
        layout=go.Layout(
            title="加速度・ジャイロセンサーデータによる飛行ログ",
            scene=dict(
                xaxis=dict(title='X (m)'),
                yaxis=dict(title='Y (m)'),
                zaxis=dict(title='Z (m)'),
                aspectmode='data'
            ),
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "再生",
                            "method": "animate",
                            "args": [None, {
                                "frame": {"duration": 50, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0}
                            }]
                        }
                    ]
                }
            ],
            sliders=[
                {
                    "active": 0,
                    "currentvalue": {"prefix": "時刻: "},
                    "pad": {"t": 50},
                    "steps": [
                        {
                            "args": [[f"frame{idx}"],
                                     {
                                         "frame": {"duration": 50, "redraw": True},
                                         "mode": "immediate",
                                         "transition": {"duration": 0}
                                     }],
                            "label": f"{time[idx]:.2f}",
                            "method": "animate"
                        }
                        for idx in indices
                    ]
                }
            ]
        ),
        frames=frames
    )
    
    # 凡例の表示設定
    fig.update_layout(
        legend=dict(
            title="凡例",
            orientation="h",
            x=0.5,
            y=1.15,
            xanchor="center",
            font=dict(size=14, color="black"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=2
        )
    )
    
    # 作成した図を表示
    fig.show()

if __name__ == "__main__":
    main()
