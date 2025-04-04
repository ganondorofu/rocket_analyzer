# coding: utf-8
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import cumulative_trapezoid
import math

# オイラー角（roll, pitch, yaw）から回転行列を計算する関数
# この関数は、与えられたroll, pitch, yaw（ラジアン）に基づき、各軸の回転行列を生成し
# それらを掛け合わせることで最終的な回転行列（3×3行列）を返します。
def euler_to_rotation_matrix(roll, pitch, yaw):
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
    # 3軸の回転行列を掛け合わせて最終的な回転行列を得る
    R = R_z @ R_y @ R_x
    return R

# メインの処理を行う関数
def main():
    # CSVファイルからセンサーデータを読み込む
    # ※ このCSVは、事前に記録されたMPU6050などのセンサーデータが含まれているものとする
    df = pd.read_csv("sensor_data28.csv")
    
    # 'Time_s' カラムが存在すればその値を使用、なければ dt=0.01 秒でタイムスタンプを生成する
    if 'Time_s' in df.columns:
        time = df['Time_s'].values
    else:
        dt = 0.01
        time = np.arange(len(df)) * dt

    # 不要なカラム（例えばGNSSなどの座標データ）を除外する
    columns_to_exclude = ['Lat', 'Lng', 'Alt_m', 'Fix', 'Sats']
    data = df.drop(columns=columns_to_exclude, errors='ignore')
    
    # 加速度センサーおよびジャイロセンサーのカラム名を抽出する
    # センサーデータが3軸分あるかをチェックする
    acc_columns = [col for col in data.columns if 'Acc' in col or 'acc' in col]
    gyro_columns = [col for col in data.columns if 'Gyro' in col or 'gyro' in col]
    if len(acc_columns) < 3:
        raise ValueError("加速度センサーのカラムが3つ以上ありません")
    if len(gyro_columns) < 3:
        raise ValueError("ジャイロセンサーのカラムが3つ以上ありません")
    
    # 最初の3カラムのみを使用（各軸X, Y, Z）
    acc_columns = acc_columns[:3]
    gyro_columns = gyro_columns[:3]
    
    # 加速度データを取得し、単位を g から m/s^2 に変換する
    # （1g = 9.81 m/s^2）
    acc_data = data[acc_columns].to_numpy() * 9.81  
    
    # ジャイロデータ（単位: deg/s）を取得し、ラジアンに変換する
    gyro_data = data[gyro_columns].to_numpy()
    gyro_data_rad = np.deg2rad(gyro_data)
    
    # ジャイロデータからオイラー角（roll, pitch, yaw）を、タイムスタンプを考慮して積分する
    # cumulative_trapezoid関数は、与えられた時刻間隔（x=time）に基づいて正確に積分を行います
    attitude = cumulative_trapezoid(gyro_data_rad, x=time, initial=0, axis=0)
    
    N = len(time)
    # 各時刻のローカル加速度データをグローバル座標系に変換するための配列を用意
    global_acc = np.zeros_like(acc_data)
    # 各時刻について、ジャイロで得たオイラー角をもとに回転行列を計算し、加速度データを変換する
    for i in range(N):
        roll = attitude[i, 0]
        pitch = attitude[i, 1]
        yaw = attitude[i, 2]
        R_mat = euler_to_rotation_matrix(roll, pitch, yaw)
        # 変換後のグローバル加速度は、回転行列とローカル加速度の積として求める
        global_acc[i] = R_mat @ acc_data[i]
    # 重力加速度（9.81 m/s^2）が常にZ軸方向に作用するため、グローバル加速度のZ軸成分から重力を除去する
    global_acc[:, 2] = global_acc[:, 2] - 9.81
    
    # 各時刻のグローバル加速度の大きさ（ノルム）を計算
    # この値は、アニメーションのカラーグラデーションやホバーテキストに利用する
    acc_magnitude = np.linalg.norm(global_acc, axis=1)
    
    # 加速度データから台形公式により速度を計算（時間軸を考慮）
    velocity = cumulative_trapezoid(global_acc, x=time, initial=0, axis=0)
    # さらに、速度の積分により位置を計算
    position = cumulative_trapezoid(velocity, x=time, initial=0, axis=0)
    
    # アニメーション用のフレームを作成するためのインデックスを決定
    # データ行数が50行未満の場合は全行を使用し、十分なフレームが得られない場合のエラーを回避する
    if N < 50:
        indices = list(range(N))
    else:
        max_frames = 100
        step = max(1, N // max_frames)
        indices = list(range(0, N, step))
        if indices[-1] != N - 1:
            indices.append(N - 1)
    
    frames = []
    axis_length = 1.0  # 局所座標軸（センサのローカル座標）の長さ（単位：メートル）
    
    # 飛行軌跡（全位置の軌道）のトレースを作成
    # カラーは各時刻の加速度の大きさにより変化し、ホバー時に各座標と加速度が表示される
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
    
    # 各フレームごとに、現在位置およびセンサの局所座標軸（X, Y, Z軸）を表示するフレームを作成
    for idx in indices:
        pos_current = position[idx]
        roll = attitude[idx, 0]
        pitch = attitude[idx, 1]
        yaw = attitude[idx, 2]
        # 現在のオイラー角に基づく回転行列を計算
        R_mat = euler_to_rotation_matrix(roll, pitch, yaw)
        # 各軸の終点は、現在位置に回転行列を掛けた単位ベクトルを加えることで求める
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
        # 局所X軸のトレース
        x_axis_trace = go.Scatter3d(
            x=[pos_current[0], x_axis_end[0]],
            y=[pos_current[1], x_axis_end[1]],
            z=[pos_current[2], x_axis_end[2]],
            mode='lines',
            line=dict(color='red', width=5),
            name='X軸'
        )
        # 局所Y軸のトレース
        y_axis_trace = go.Scatter3d(
            x=[pos_current[0], y_axis_end[0]],
            y=[pos_current[1], y_axis_end[1]],
            z=[pos_current[2], y_axis_end[2]],
            mode='lines',
            line=dict(color='green', width=5),
            name='Y軸'
        )
        # 局所Z軸のトレース
        z_axis_trace = go.Scatter3d(
            x=[pos_current[0], z_axis_end[0]],
            y=[pos_current[1], z_axis_end[1]],
            z=[pos_current[2], z_axis_end[2]],
            mode='lines',
            line=dict(color='orange', width=5),
            name='Z軸'
        )
        # 各フレームのデータとしてまとめ、フレームリストに追加
        frame_data = [flight_path_trace, current_pos_trace, x_axis_trace, y_axis_trace, z_axis_trace]
        frames.append(go.Frame(data=frame_data, name=f"frame{idx}"))
    
    # 初期フレーム（アニメーション開始時）のデータを作成
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
    
    # 凡例（レジェンド）の表示設定（中央上部に配置し、フォントサイズや背景色、枠線を調整）
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
    
    # 作成した図を表示する
    fig.show()

if __name__ == "__main__":
    main()
