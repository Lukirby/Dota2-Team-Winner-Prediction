import pandas as pd
import warnings
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier

seed = 10

def get_dataset():

    featureset_path = "../dataset/mlcourse-dota2-win-prediction/train_features.csv"
    targetset_path = "../dataset/mlcourse-dota2-win-prediction/train_targets.csv"

    df = pd.read_csv(featureset_path)
    target = pd.read_csv(targetset_path)

    #print("Features: ",df.columns,"\n")
    #print("Target Columns: ",target.columns,"\n")

    column_to_drop = ["lobby_type","chat_len","game_mode","match_id_hash"] # "match_id_hash","objectives_len"
    
    filter = "(game_mode == 2 or game_mode == 22) and game_time > 0" # 2 standard ranked or 22 captain mode

    df = df.query(filter)

    df = df.drop(labels=column_to_drop,axis=1)

    tf_toreplace = ["r1_teamfight_participation",
                    "r2_teamfight_participation",
                    "r3_teamfight_participation",
                    "r4_teamfight_participation",
                    "r5_teamfight_participation",
                    "d1_teamfight_participation",
                    "d2_teamfight_participation",
                    "d3_teamfight_participation",
                    "d4_teamfight_participation",
                    "d5_teamfight_participation"]

    for label in tf_toreplace:
        df.loc[df[label] > 1.0, label] = 1


    print("Filtering Df: ", filter, "\n")

    print("Dropped: ",column_to_drop,"\n")

    print("Dataframe Shape: ",df.shape,"\n")
    
    target = target.loc[df.index]
    print(f"Target shape: {target.shape}")
    return df,target


def get_hero_id_labels(df: pd.DataFrame) -> list[str]:
    hero_id_labels = [s for s in df.columns if s.endswith('_hero_id')]
    print("Hero Id Labels:",hero_id_labels,"\n")
    return hero_id_labels

def get_single_hero_labels(df: pd.DataFrame) -> list[str]:
    single_hero_labels = [s for s in df.columns if re.match(r"^(d|r)\d",s)]
    print("Single Player Labels:",single_hero_labels,"\n")
    return single_hero_labels

def drop_heros_labels(df:pd.DataFrame) -> pd.DataFrame:
    hero_id_labels = get_hero_id_labels(df)
    if (len(hero_id_labels) == 0):
        for label in df.columns:
            if re.match(r"^(d|r)_\d+$", label):  #regex: r_1 d_2 r_124 etc... 
                df = df.drop(label,axis=1)
            elif re.match(r"^(d|r)\d_heroid\d+$",label):      #regex: r1_hero_id_12 d3_hero_id_101 ecc..
                df = df.drop(label,axis=1)
    else:
        df = df.drop(labels=hero_id_labels,axis=1)

    print("Dropped Dataframe Shape:",df.shape)

    return df


def playerstats_playerheros_transform(df: pd.DataFrame):

    features_toonehot = ["r1_hero_id",
                         "r2_hero_id",
                         "r3_hero_id",
                         "r4_hero_id",
                         "r5_hero_id",
                         "d1_hero_id",
                         "d2_hero_id",
                         "d3_hero_id",
                         "d4_hero_id",
                         "d5_hero_id"]
    df = pd.get_dummies(df,columns=features_toonehot)

    #target = target.loc[df.index]
    #print(target.shape)
    #df = df.drop('match_id_hash',axis=1)

    return df

def playerstats_teamheros_transform(df: pd.DataFrame):
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
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

    for hero_id in hero_id_set_tot:
        df[f"r_{hero_id}"] = 0
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

    #target = target.loc[df.index]
    #print(target.shape)
    

    return df

def teamstats_teamheros_transform(df: pd.DataFrame):
    #we handle PerformanceWarning by doing the copy of the dataframe, this ignore is for quality of outputs
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
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

    df = df.drop(labels=hero_id_labels,axis=1) #removed ri_hero_id and di_hero_id

    #print("Dataframe Shape:",df.shape,"\n")

    #print(df.iloc[0]["match_id_hash"])
    #print(df.iloc[0][df.iloc[0] == 1][-11:])

    print("NaN Count: ",pd.isna(df).sum().sum(),"\n")

    df = df.copy()

    """ i = 0
    for v in df['d_32']:
        if v == 1:
            i += 1
    print(f"Total: {i}") """

    single_hero_labels = get_single_hero_labels(df)
    single_hero_labels2 = single_hero_labels.copy()
    for label in single_hero_labels:
        if re.match(r".*(_x|_y)$",label):
            single_hero_labels2.remove(label)
            continue
        new_label = label[0]+label[2:] #r1_gold -> r_gold
        if not (new_label in df.columns):
            df[new_label] = df[label]
        else:
            df[new_label] += df[label]
    single_hero_labels = single_hero_labels2.copy()
    df = df.drop(labels=single_hero_labels,axis=1).copy()
    #print("New Dataframe Colums:",df.columns,"\n")
    print("New Dataframe Shape:",df.shape,"\n")

    #print(df.query("d_firstblood_claimed == 0 and r_firstblood_claimed == 0").shape)

    #for label in df.columns:
    #    if re.match(r"^(d|r)_\d*$",label): #regex to drop all d_numbers to drop heroes
    #        df = df.drop(label,axis=1)
    #print(df.shape)

    #target = target.loc[df.index]
    #print(target.shape)
    

    return df

def team_mean_position_transform (df: pd.DataFrame):
    labels_radiant_x = ["r1_x", "r2_x", "r3_x", "r4_x", "r5_x"]
    labels_radiant_y = ["r1_y", "r2_y", "r3_y", "r4_y", "r5_y"]
    labels_dire_x = ["d1_x", "d2_x", "d3_x", "d4_x", "d5_x"]
    labels_dire_y = ["d1_y", "d2_y", "d3_y", "d4_y", "d5_y"]

    #calculate average x and y for Radiant team
    df['radiant_avg_x'] = df[labels_radiant_x].mean(axis=1)
    df['radiant_avg_y'] = df[labels_radiant_y].mean(axis=1)

    #calculate average x and y for Dire team
    df['dire_avg_x'] = df[labels_dire_x].mean(axis=1)
    df['dire_avg_y'] = df[labels_dire_y].mean(axis=1)

    return df

def team_weighted_mean_position_transform(df: pd.DataFrame): 

    labels_radiant_x = ["r1_x", "r2_x", "r3_x", "r4_x", "r5_x"]
    labels_radiant_y = ["r1_y", "r2_y", "r3_y", "r4_y", "r5_y"]
    labels_dire_x = ["d1_x", "d2_x", "d3_x", "d4_x", "d5_x"]
    labels_dire_y = ["d1_y", "d2_y", "d3_y", "d4_y", "d5_y"]

    df_Weighted = df.copy(deep=True)
    df_Weighted  = get_average_distances(df_Weighted) 

    distances_radiant = ["distance_r1", "distance_r2", "distance_r3", "distance_r4", "distance_r5"]
    distances_dire = ["distance_d1", "distance_d2", "distance_d3", "distance_d4", "distance_d5"]
    #calculate weights as the inverse of distances
    weights_radiant = 1 / df_Weighted[distances_radiant]
    weights_dire = 1 / df_Weighted[distances_dire]
    
    #calculate average x and y for Radiant team
    #df['radiant_Weighted_avg_x'] = (df[labels_radiant_x] * weights_radiant).sum(axis=1) / weights_radiant.sum(axis=1)
    #df['radiant_Weighted_avg_y'] = (df[labels_radiant_y] * weights_radiant).sum(axis=1) / weights_radiant.sum(axis=1)
    #calculate average x and y for Dire team
    #df['dire_Weighted_avg_x'] = (df[labels_dire_x] * weights_dire).sum(axis=1) / weights_dire.sum(axis=1)
    #df['dire_Weighted_avg_y'] = (df[labels_dire_y] * weights_dire).sum(axis=1) / weights_dire.sum(axis=1)
    #initialize weighted average columns
    df['radiant_Weighted_avg_x'] = 0
    df['radiant_Weighted_avg_y'] = 0
    df['dire_Weighted_avg_x'] = 0
    df['dire_Weighted_avg_y'] = 0
    for i in range(5):
        #calculate weighted average x and y for Radiant team
        df['radiant_Weighted_avg_x'] += df[labels_radiant_x[i]] * weights_radiant.iloc[:, i]
        df['radiant_Weighted_avg_y'] += df[labels_radiant_y[i]] * weights_radiant.iloc[:, i]

        #calculate weighted average x and y for Dire team
        df['dire_Weighted_avg_x'] += df[labels_dire_x[i]] * weights_dire.iloc[:, i]
        df['dire_Weighted_avg_y'] += df[labels_dire_y[i]] * weights_dire.iloc[:, i]

    #normalize by the sum of weights
    df['radiant_Weighted_avg_x'] /= weights_radiant.sum(axis=1)
    df['radiant_Weighted_avg_y'] /= weights_radiant.sum(axis=1)
    df['dire_Weighted_avg_x'] /= weights_dire.sum(axis=1)
    df['dire_Weighted_avg_y'] /= weights_dire.sum(axis=1)
    return df


#this is ok, tested
def get_average_distances(df: pd.DataFrame):
    labels_radiant_x = ["r1_x", "r2_x", "r3_x", "r4_x", "r5_x"]
    labels_radiant_y = ["r1_y", "r2_y", "r3_y", "r4_y", "r5_y"]
    labels_dire_x = ["d1_x", "d2_x", "d3_x", "d4_x", "d5_x"]
    labels_dire_y = ["d1_y", "d2_y", "d3_y", "d4_y", "d5_y"]

    radiant_distances = calculate_distances(df, labels_radiant_x, labels_radiant_y)
    dire_distances = calculate_distances(df, labels_dire_x, labels_dire_y)

    for label in radiant_distances:
        truncated_label = label[:-2]
        df[f'distance_{truncated_label}'] = radiant_distances[label]

    for label in dire_distances:
        truncated_label = label[:-2]
        df[f'distance_{truncated_label}'] = dire_distances[label]

    return df
#this is ok, tested
def calculate_distances(df: pd.DataFrame, x_labels, y_labels):
    distances = {label: [] for label in x_labels}
    for i in range(len(x_labels)):
        for j in range(len(x_labels)):
            if i != j:
                dist = np.sqrt((df[x_labels[i]] - df[x_labels[j]])**2 + (df[y_labels[i]] - df[y_labels[j]])**2)
                distances[x_labels[i]].append(dist)
    return {label: np.mean(distances[label], axis=0) for label in distances}


def feature_selection_transform(df: pd.DataFrame,target: pd.DataFrame, threshold: float) -> pd.DataFrame:
    feature_selector = RandomForestClassifier(max_depth=10,class_weight="balanced",random_state=seed)

    feature_selector.fit(df,target)

    feature_importance = {
        name: value 
        for name,value in zip(feature_selector.feature_names_in_,feature_selector.feature_importances_)
    }

    feature_importance = dict(reversed(sorted(feature_importance.items(), key=lambda item: item[1])))
    feature_names = list(feature_importance.keys())

    n_selected_features = np.sum(np.array(list(feature_importance.values())) > threshold)

    df_reduced = df[feature_names[:n_selected_features]]
    print("Shape Tranformation:\n",df.shape,"->", df_reduced.shape)

    return df_reduced    