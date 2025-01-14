import pandas as pd
import numpy as np
import re
import warnings

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

    tf_toreplace = ["r1_teamfight_participation", "r2_teamfight_participation","r3_teamfight_participation", "r4_teamfight_participation",
                    "r5_teamfight_participation","d1_teamfight_participation","d2_teamfight_participation","d3_teamfight_participation",
                    "d4_teamfight_participation",  "d5_teamfight_participation"]
    
    for label in tf_toreplace:
        df.loc[df[label] > 1.0, label] = 1
    
    #a = df.loc[df["match_id_hash"] == "a400b8f29dece5f4d266f49f1ae2e98a"] #hash where nothing change = a400b8f29dece5f4d266f49f1ae2e98a ; hash where something change = 8e0ad8cbcf5a87c451e5e1e07596c443
    #print(a[tf_toreplace])

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

def drop_heros_labels(df:pd.DataFrame) -> pd.DataFrame:
    hero_id_labels = get_hero_id_labels(df)
    if (len(hero_id_labels) == 0):
        for label in df.columns:
            if re.match(r"^(d|r)_\d*$",label):  #regex: r_1 d_2 ecc...
                df = df.drop(label,axis=1)
            elif re.match(r"^(d|r)\d_hero_id_\d*$",label):      #regex: r1_hero_id_12 d3_hero_id_101 ecc..
                df = df.drop(label,axis=1)
    else:
        df = df.drop(labels=hero_id_labels,axis=1)
    
    print("Dropped Dataframe Shape:",df.shape)

    return df


def playerstats_playerheros_transform(df: pd.DataFrame, target: pd.DataFrame):

    features_toonehot = ["r1_hero_id","r2_hero_id","r3_hero_id","r4_hero_id","r5_hero_id","d1_hero_id","d2_hero_id","d3_hero_id","d4_hero_id","d5_hero_id"]
    df = pd.get_dummies(df,columns=features_toonehot)

    target = target.loc[df.index]
    print(target.shape)
    df = df.drop('match_id_hash',axis=1)

    return df,target

def playerstats_teamheros_transform(df: pd.DataFrame, target: pd.DataFrame):

    hero_id_labels = get_hero_id_labels(df)

    hero_id_set = {i: set() for i in range(len(hero_id_labels))}

    hero_id_set_tot = set()

    for n,label in enumerate(hero_id_labels):
        for id in df[label]:
            hero_id_set[n].add(id)

    for i in range(len(hero_id_labels)):
        #print(f"{i}. {len(hero_id_set[i])}")
        hero_id_set_tot = hero_id_set_tot.union(hero_id_set[i])

    print("Numbers of Heros: ",len(hero_id_set_tot),"\n")

    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    for hero_id in hero_id_set_tot:
        df[f"r_{hero_id}"] = (
            (df["r1_hero_id"] == hero_id) | 
            (df["r2_hero_id"] == hero_id) |
            (df["r3_hero_id"] == hero_id) |
            (df["r4_hero_id"] == hero_id) |
            (df["r5_hero_id"] == hero_id)
        ).astype(int)
        df[f"d_{hero_id}"] = (
            (df["d1_hero_id"] == hero_id) | 
            (df["d2_hero_id"] == hero_id) |
            (df["d3_hero_id"] == hero_id) |
            (df["d4_hero_id"] == hero_id) |
            (df["d5_hero_id"] == hero_id)
        ).astype(int)

    warnings.simplefilter(action="default",category=pd.errors.PerformanceWarning)

    df = df.drop(labels=hero_id_labels,axis=1) #removed ri_hero_id and di_hero_id

    print("Dataframe Shape:",df.shape,"\n")

    #print(df.iloc[0]["match_id_hash"])
    #print(df.iloc[0][df.iloc[0] == 1][-11:])

    print("NaN Count: ",pd.isna(df).sum().sum(),"\n")

    df = df.copy()

    """ i = 0
    for v in df['d_32']:
        if v == 1:
            i += 1
    print(f"Total: {i}") """

    target = target.loc[df.index]
    print(target.shape)
    df = df.drop('match_id_hash',axis=1)

    return df,target

def teamstats_teamheros_transform(df: pd.DataFrame, target: pd.DataFrame):

    #region team heroes grouping

    hero_id_labels = get_hero_id_labels(df)

    hero_id_set = {i: set() for i in range(len(hero_id_labels))}

    hero_id_set_tot = set()

    for n,label in enumerate(hero_id_labels):
        for id in df[label]:
            hero_id_set[n].add(id)

    for i in range(len(hero_id_labels)):
        #print(f"{i}. {len(hero_id_set[i])}")
        hero_id_set_tot = hero_id_set_tot.union(hero_id_set[i])

    print("Numbers of Heros: ",len(hero_id_set_tot),"\n")

    # r_100 = r1_hero_id == 100 or r2_hero_id == 100 or r3_hero_id == 100 ... 
    for hero_id in hero_id_set_tot:
        df[f"r_{hero_id}"] = (
            (df["r1_hero_id"] == hero_id) | 
            (df["r2_hero_id"] == hero_id) |
            (df["r3_hero_id"] == hero_id) |
            (df["r4_hero_id"] == hero_id) |
            (df["r5_hero_id"] == hero_id)
        ).astype(int)
        df[f"d_{hero_id}"] = (
            (df["d1_hero_id"] == hero_id) | 
            (df["d2_hero_id"] == hero_id) |
            (df["d3_hero_id"] == hero_id) |
            (df["d4_hero_id"] == hero_id) |
            (df["d5_hero_id"] == hero_id)
        ).astype(int)

    #endregion 

    df = df.drop(labels=hero_id_labels,axis=1) #removed ri_hero_id and di_hero_id

    print("Dataframe Shape:",df.shape,"\n")

    #print(df.iloc[0]["match_id_hash"])
    #print(df.iloc[0][df.iloc[0] == 1][-11:])

    print("NaN Count: ",pd.isna(df).sum().sum(),"\n")

    df = df.copy()

    """ i = 0
    for v in df['d_32']:
        if v == 1:
            i += 1
    print(f"Total: {i}") """

    #region team stats grouping
    single_hero_labels = get_single_hero_labels(df)
    single_hero_labels2 = single_hero_labels.copy()
    for label in single_hero_labels:
        if re.match(r".*(_x|_y)$",label):
            single_hero_labels2.remove(label)
            continue
        new_label = label[0]+label[2:]              #r1_gold -> r_gold
        if not (new_label in df.columns):
            df[new_label] = df[label]
        else:
            df[new_label] += df[label]
    single_hero_labels = single_hero_labels2.copy()
    df = df.drop(labels=single_hero_labels,axis=1).copy()
    print("New Dataframe Colums:",df.columns,"\n")
    print("New Dataframe Shape:",df.shape,"\n")

    #print(df.query("d_firstblood_claimed == 0 and r_firstblood_claimed == 0").shape)

    #endregion

    target = target.loc[df.index]
    print(target.shape)
    df = df.drop('match_id_hash',axis=1)

    return df,target
