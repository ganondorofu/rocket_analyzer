# coding: utf-8
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import cumulative_trapezoid
import math

# オイラー角（roll, pitch, yaw）から回転行列を計算する関数
def euler_to_rotation_matrix(roll, pitch, yaw):
    # X軸回りの回転行列
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    # Y軸回りの回転行列
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    # Z軸回りの回転行列
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    # 3軸の回転行列を掛け合わせて最終的な回転行列を得る
    R = R_z @ R_y @ R_x
    return R

# メインの処理を行う関数
def main():
    # CSVファイルからセンサーデータを読み込む
    df = pd.read_csv("sensor_data50.csv")
    
    # 'Time_s' カラムが存在すればその値を使用、なければ dt=0.01 秒でタイムスタンプを生成
    if 'Time_s' in df.columns:
        time = df['Time_s'].values
        dt = np.mean(np.diff(time))
    else:
        dt = 0.01
        time = np.arange(len(df)) * dt

    # 不要なカラムを除外する
    columns_to_exclude = ['Lat', 'Lng', 'Alt_m', 'Fix', 'Sats']
    data = df.drop(columns=columns_to_exclude, errors='ignore')
    
    # 加速度センサーおよびジャイロセンサーのカラム名を抽出
    acc_columns = [col for col in data.columns if 'Acc' in col or 'acc' in col]
    gyro_columns = [col for col in data.columns if 'Gyro' in col or 'gyro' in col]
    
    # 加速度センサー、ジャイロセンサーともに3軸のデータがあるかチェック
    if len(acc_columns) < 3:
        raise ValueError("加速度センサーのカラムが3つ以上ありません")
    if len(gyro_columns) < 3:
        raise ValueError("ジャイロセンサーのカラムが3つ以上ありません")
    
    # 先頭の3カラムを使用
    acc_columns = acc_columns[:3]
    gyro_columns = gyro_columns[:3]
    
    # 加速度データを取得し、単位を g から m/s^2 に変換
    acc_data = data[acc_columns].to_numpy() * 9.81  
    # ジャイロデータ（単位: deg/s）を取得
    gyro_data = data[gyro_columns].to_numpy()   
    
    # ジャイロデータをラジアンに変換
    gyro_data_rad = np.deg2rad(gyro_data)
    # 各時刻でのオイラー角（roll, pitch, yaw）を積分により計算
    attitude = np.cumsum(gyro_data_rad * dt, axis=0)
    
    N = len(time)
    global_acc = np.zeros_like(acc_data)
    # 各時刻における加速度をグローバル座標系に変換
    for i in range(N):
        roll = attitude[i, 0]
        pitch = attitude[i, 1]
        yaw = attitude[i, 2]
        R_mat = euler_to_rotation_matrix(roll, pitch, yaw)
        global_acc[i] = R_mat @ acc_data[i]
    # 重力加速度を除去（Z軸方向）
    global_acc[:, 2] = global_acc[:, 2] - 9.81
    
    # 各時刻のグローバル加速度の大きさを計算（カラーグラデーションおよびホバーテキストの指標として利用）
    acc_magnitude = np.linalg.norm(global_acc, axis=1)
    
    # 加速度の積分により速度、さらに速度の積分により位置を計算（台形公式）
    velocity = cumulative_trapezoid(global_acc, dx=dt, initial=0, axis=0)
    position = cumulative_trapezoid(velocity, dx=dt, initial=0, axis=0)
    
    # アニメーションで使用するフレーム数の上限を設定
    max_frames = 100
    step = max(1, N // max_frames)
    indices = list(range(0, N, step))
    if indices[-1] != N - 1:
        indices.append(N - 1)
    
    frames = []
    axis_length = 1.0  # 局所座標軸の長さ（メートル）
    
    # 飛行軌跡のトレースを作成
    # customdata と hovertemplate を設定して、カーソルオーバー時に加速度の値も表示する
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
    
    # 各フレームごとに現在位置と局所座標軸を表示するフレームを作成
    for idx in indices:
        pos_current = position[idx]
        roll = attitude[idx, 0]
        pitch = attitude[idx, 1]
        yaw = attitude[idx, 2]
        R_mat = euler_to_rotation_matrix(roll, pitch, yaw)
        # 局所座標軸の終点を計算
        x_axis_end = pos_current + R_mat @ np.array([axis_length, 0, 0])
        y_axis_end = pos_current + R_mat @ np.array([0, axis_length, 0])
        z_axis_end = pos_current + R_mat @ np.array([0, 0, axis_length])
        
        # 現在位置を示すマーカー
        current_pos_trace = go.Scatter3d(
            x=[pos_current[0]],
            y=[pos_current[1]],
            z=[pos_current[2]],
            mode='markers',
            marker=dict(size=5, color='red'),
            name='現在位置'
        )
        # 局所X軸の表示
        x_axis_trace = go.Scatter3d(
            x=[pos_current[0], x_axis_end[0]],
            y=[pos_current[1], x_axis_end[1]],
            z=[pos_current[2], x_axis_end[2]],
            mode='lines',
            line=dict(color='red', width=5),
            name='X軸'
        )
        # 局所Y軸の表示
        y_axis_trace = go.Scatter3d(
            x=[pos_current[0], y_axis_end[0]],
            y=[pos_current[1], y_axis_end[1]],
            z=[pos_current[2], y_axis_end[2]],
            mode='lines',
            line=dict(color='green', width=5),
            name='Y軸'
        )
        # 局所Z軸の表示
        z_axis_trace = go.Scatter3d(
            x=[pos_current[0], z_axis_end[0]],
            y=[pos_current[1], z_axis_end[1]],
            z=[pos_current[2], z_axis_end[2]],
            mode='lines',
            line=dict(color='orange', width=5),
            name='Z軸'
        )
        # 各フレームのデータとして追加
        frame_data = [flight_path_trace, current_pos_trace, x_axis_trace, y_axis_trace, z_axis_trace]
        frames.append(go.Frame(data=frame_data, name=f"frame{idx}"))
    
    # 初期フレームのデータを作成
    init_idx = indices[0]
    pos_init = position[init_idx]
    roll_init = attitude[init_idx, 0]
    pitch_init = attitude[init_idx, 1]
    yaw_init = attitude[init_idx, 2]
    R_init = euler_to_rotation_matrix(roll_init, pitch_init, yaw_init)
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
    
    # 図のレイアウトおよびアニメーションの設定
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
                            "args": [None, {"frame": {"duration": 50, "redraw": True},
                                            "fromcurrent": True, "transition": {"duration": 0}}]
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
                                     {"frame": {"duration": 50, "redraw": True},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
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
    
    # 凡例の表示設定を改善（中央上部に配置し、フォントサイズや背景色、枠線を調整）
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
