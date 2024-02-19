import os
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import cv2
from TrackCollection import generate_tracking_data
from TrackProcessing import get_track_data

def get_user_input_defensive_team(teams_list):
    return teams_list[input("type 0 for " + teams_list[0] + " or 1 for " + teams_list[1])]

def calculate_frame_spacing(df, frame_num, team_abbrev):
    frame_df = df[df["FRAME_NUM"] == frame_num]
    team_df = df[df["TEAM_ABBREV"] == team_abbrev]
    coords = team_df[["X", "Y"]].to_numpy()

    return ConvexHull(coords).volume

def calculate_possession_spacing(track_data, defensive_team):
    frame_nums = [float(i) for i in np.unique(track_data)]
    spacing = [calculate_frame_spacing(track_data, i, defensive_team) for i in frame_nums]

    return np.trapz(y=spacing, x=frame_nums)

def collect_spacing_metrics(dir, restrict_ten_longest=False):
    titles = []
    spacing_metrics = []

    for filename in os.listdir(dir):
        json_out = generate_tracking_data(filename, dir)
        processed_data = get_track_data(json_out, filename, restrict_ten_longest=restrict_ten_longest)
        defensive_team = get_user_input_defensive_team(list(np.unique(processed_data["TEAM_ABBREV"])))

        spacing = calculate_possession_spacing(processed_data, defensive_team)

        titles.append(filename)
        spacing_metrics.append(spacing)

    return pd.DataFrame({"POSSESSION": titles, "SPACING": spacing_metrics})

#spacing_data = collect_spacing_metrics("Footage/Cavs/Donovan_Mitchell/")
#spacing_data.to_csv("DONOVAN_MITCHELL_SPACING.csv")












