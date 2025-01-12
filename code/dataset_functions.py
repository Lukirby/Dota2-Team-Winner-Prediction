import pandas as pd
import numpy as np
import re

def get_dataset():

    featureset_path = "../dataset/mlcourse-dota2-win-prediction/train_features.csv"
    targetset_path = "../dataset/mlcourse-dota2-win-prediction/train_targets.csv"

    df = pd.read_csv(featureset_path)
    target = pd.read_csv(targetset_path)

    print("Features: ",df.columns,"\n")
    print("Target Columns: ",target.columns,"\n")

    column_to_drop = ["lobby_type","chat_len","game_mode"] # "match_id_hash","objectives_len"

    filter = "(game_mode == 2 or game_mode == 22) and game_time > 0" # 2 standard ranked or 22 captain mode

    df = df.query(filter)

    df = df.drop(labels=column_to_drop,axis=1)


    print("Filtering Df: ", filter, "\n")

    print("Dropped: ",column_to_drop,"\n")

    print("Dataframe Shape: ",df.shape,"\n")

    return df,target


def get_hero_id_labels(df: pd.DataFrame) -> list[str]:
    hero_id_labels = [s for s in df.columns if s.endswith('_hero_id')]
    print("Hero Id Labels:",hero_id_labels,"\n")
    return hero_id_labels

def get_single_hero_labels(df: pd.DataFrame) -> list[str]:
    single_hero_labels = [s for s in df.columns if re.match(r"^(d|r)\d",s)]
    print("Single Hero Labels:",single_hero_labels,"\n")
    return single_hero_labels
