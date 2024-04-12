import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import tqdm
from datasets import Dataset, DatasetDict
from tqdm.contrib.logging import logging_redirect_tqdm

system_message = """**Autonomous Driving Planner**
Role: You are the brain of an autonomous vehicle. Plan a safe 6-second driving target. Avoid collisions with other objects.

Context
- Coordinates: X-axis is forward, and Y-axis is left relative to the ego vehicle. The ego vehicle is at point (0,0).
- Objective: Create a 6-second destination point and destination velocity vector.

Inputs
1. Perception: Info about surrounding objects, their positions, their velocities, and their headings.
2. Ego-State: Your current state including velocity.

Task
- Target Prediction: Determine a safe and feasible target point 6 seconds into the future.

Output
- Target position is (x, y): ...
- Target velocity is (vx, vy): ...
- Target heading is ...
"""


def convert_scenario_data(df: pd.DataFrame, past_horizon, fut_horizon: int, sampling_freq: int) -> List[Dict]:
    """
    Convert scenario data into conversational format

    example user message:
    Ego-States:
    - Velocity (vx, vy): (0.00, 0.00)
    Perception:
    - Vehicle at (x, y): (0.00, 0.00), (vx, vy): (0.00, 0.00)
    - Vehicle at (x, y): (0.00, 0.00), (vx, vy): (0.00, 0.00)
    - Vehicle at (x, y): (0.00, 0.00), (vx, vy): (0.00, 0.00)

    example assistant message:
    Target position is (x, y): (0.00, 0.00)
    Target velocity is (vx, vy): (0.00, 0.00)

    """
    t_curr = past_horizon
    t_fut = past_horizon + fut_horizon

    user_msg = system_message + "\n"
    # transform object positions, headings, velocities into AV frame at t_curr
    df_curr = df.loc[df['timestep'] == t_curr]
    agent_ids = list(df_curr['track_id'].unique())
    df_fut = df.loc[df['timestep'] == t_fut]
    av_pos_curr = df_curr.loc[df_curr['track_id']
                              == 'AV', ['position_x', 'position_y']].values.squeeze()
    av_heading_curr = df_curr.loc[df_curr['track_id']
                                  == 'AV', 'heading'].values.squeeze()
    # TODO vectorize the timestep iteration
    rot = np.array([[np.cos(av_heading_curr), -np.sin(av_heading_curr)],
                    [np.sin(av_heading_curr), np.cos(av_heading_curr)]])  # [2, 2, time_steps]
    pos_curr = np.dot(
        rot.T, df_curr[['position_x', 'position_y']].values.T - av_pos_curr[:, None]).T
    pos_fut = np.dot(
        rot.T, df_fut[['position_x', 'position_y']].values.T - av_pos_curr[:, None]).T
    vel_curr = np.dot(
        rot.T, df_curr[['velocity_x', 'velocity_y']].values.T).T
    vel_fut = np.dot(
        rot.T, df_fut[['velocity_x', 'velocity_y']].values.T).T
    heading_curr = df_curr['heading'] - av_heading_curr
    heading_fut = df_fut['heading'] - av_heading_curr
    # wrap heading to [-pi, pi]
    heading_curr = (heading_curr + np.pi) % (2 * np.pi) - np.pi
    heading_fut = (heading_fut + np.pi) % (2 * np.pi) - np.pi
    user_msg += "Ego-States:\n"
    vx, vy = vel_curr[df_curr['track_id'] == 'AV'][0]
    user_msg += f" - Velocity (vx, vy): ({vx:.2f}, {vy:.2f})\n"
    past_traj = []
    for i in range(0, t_curr, sampling_freq):
        pos = np.dot(
            rot.T, df.loc[(df['timestep'] == i) & (df['track_id'] == 'AV'), ['position_x', 'position_y']].values.squeeze() - av_pos_curr).T
        pos = np.round(pos, 2)
        past_traj.append(tuple(pos))
    user_msg += f" - Past Trajectory (last 2 seconds): {past_traj}\n"

    user_msg += "Perception:\n"
    for i in range(len(agent_ids)):
        if agent_ids[i] == 'AV':
            continue
        x, y = pos_curr[i]
        vx, vy = vel_curr[i]
        object_name = df_curr.loc[df_curr['track_id']
                                  == agent_ids[i], 'object_type'].values[0]
        heading = heading_curr.loc[df_curr['track_id']
                                   == agent_ids[i]].values[0]
        user_msg += f" - {object_name} at position (x, y): ({x:.2f}, {y:.2f}), velocity (vx, vy): ({vx:.2f}, {vy:.2f}), heading: ({heading:.2f})\n"

    assistant_msg = ""
    target_pos = pos_fut[df_fut['track_id'] == 'AV'][0]
    assistant_msg += "Meta Action: " + \
        generate_meta_action(
            target_pos, np.linalg.norm(vel_curr[df_curr['track_id'] == 'AV'][0]), np.linalg.norm(vel_fut[df_fut['track_id'] == 'AV'][0])) + "\n"

    assistant_msg += "Target Trajectory: \n["
    for i in range(0, fut_horizon, sampling_freq):
        pos = np.dot(
            rot.T, df.loc[(df['timestep'] == t_curr+sampling_freq+i) & (df['track_id'] == 'AV'), ['position_x', 'position_y']].values.squeeze() - av_pos_curr).T
        pos = np.round(pos, 2)
        assistant_msg += f"({pos[0]:.2f}, {pos[1]:.2f}), "
    assistant_msg += "]\n"
    assistant_msg += "Target position is (x, y): ({:.2f}, {:.2f})\n".format(
        target_pos[0], target_pos[1])
    target_vel = vel_fut[df_fut['track_id'] == 'AV'][0]
    assistant_msg += "Target velocity is (vx, vy): ({:.2f}, {:.2f})\n".format(
        target_vel[0], target_vel[1])
    assistant_msg += "Target heading is {:.2f}\n".format(
        heading_fut[df_fut['track_id'] == 'AV'].values[0])
    return [{"role": "user", "content": user_msg}, {"role": "assistant", "content": assistant_msg}]


def process_split(args, split) -> Dataset:
    conversational_format = []
    root_path = os.path.join(args.root, split)
    raw_file_names = [name for name in os.listdir(
        root_path) if os.path.isdir(os.path.join(root_path, name))]
    with logging_redirect_tqdm():
        for raw_file_name in tqdm.tqdm(raw_file_names):
            df = pd.read_parquet(os.path.join(
                root_path, raw_file_name, f'scenario_{raw_file_name}.parquet'))
            scenario_converted = convert_scenario_data(
                df, args.past_horizon, args.fut_horizon, args.sampling_freq)
            conversational_format.append({"messages": scenario_converted})
    return Dataset.from_list(conversational_format)


def generate_meta_action(target_pos, ego_cur_vel, ego_end_vel):
    """
    Generate meta action based on future trajectories of ego and agents
    """
    constant_eps = 1  # m/s
    if ego_end_vel < constant_eps and ego_cur_vel < constant_eps:
        return "STOP"
    elif ego_end_vel < constant_eps:
        speed_meta = "A DECELERATION TO ZERO"
    elif np.abs(ego_end_vel-ego_cur_vel) < constant_eps:
        speed_meta = "A CONSTANT SPEED"
    elif ego_cur_vel > ego_end_vel:
        if ego_cur_vel > 2 * ego_end_vel:
            speed_meta = "A QUICK DECELERATION"
        else:
            speed_meta = "A DECELERATION"
    else:  # ego_cur_vel < ego_end_vel
        if ego_end_vel > 2 * ego_cur_vel:
            speed_meta = "A QUICK ACCELERATION"
        else:
            speed_meta = "AN ACCELERATION"
    forward_th = 2.0  # m
    lane_changing_th = 4.0  # m
    if (np.abs(target_pos[1]) < forward_th):  # little lateral movement
        behavior_meta = "MOVE FORWARD"
    else:
        if target_pos[1] > 0:  # left
            if np.abs(target_pos[1]) > lane_changing_th:
                behavior_meta = "TURN LEFT"
            else:
                behavior_meta = "CHANGE LANE TO THE LEFT"
        elif target_pos[1] < 0:  # right
            if np.abs(target_pos[1]) > lane_changing_th:
                behavior_meta = "TURN RIGHT"
            else:
                behavior_meta = "CHANGE LANE TO THE RIGHT"
        else:
            raise ValueError(f"Undefined behaviors: {target_pos}")
    return behavior_meta + " WITH " + speed_meta


def collision_detection(x1, y1, sx1, sy1, x2, y2, sx2, sy2, x_space=1.0, y_space=3.0):  # safe distance
    if (np.abs(x1-x2) < sx1+sx2+x_space) and (y2 > y1) and (y2 - y1 < sy1+sy2+y_space):  # in front of you
        return True
    else:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, default="/home/brian/Documents/School/ECE1724/P3/GPT-Driver/data/av2")
    parser.add_argument(
        "--output", type=str, default="/home/brian/Documents/School/ECE1724/P3/GPT-Driver/data/av2_cot")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite output directory if it exists")
    parser.add_argument(
        "--fut_horizon", type=int, default=60, help="Number of timesteps to predict in the future. 10 hz, default 60 (6 seconds)")
    parser.add_argument(
        "--past_horizon", type=int, default=20, help="Number of timesteps to consider in the past. Default 20 (2 seconds)")
    parser.add_argument(
        "--sampling_freq", type=int, default=5, help="Sampling frequency of the dataset. Default 5 (2 hz)")
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    elif not args.force:
        raise ValueError(f"{args.output} already exists")
    else:
        print(f"Overwriting {args.output}")
    dataset_dict = DatasetDict()
    for split in ["train", "val"]:
        dataset_dict[split] = process_split(args, split)
    dataset_dict.save_to_disk(args.output)


if __name__ == "__main__":
    main()
