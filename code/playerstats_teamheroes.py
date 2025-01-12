import pandas as pd
import numpy as np
from dataset_functions import *
import re

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

    single_hero_labels = get_single_hero_labels(df)
    single_hero_labels2 = single_hero_labels.copy()
    for label in single_hero_labels:
        if re.match(r".*(_x|_y)$",label):
            single_hero_labels2.remove(label)
            continue
        new_label = label[0]+label[2:]
        if not (new_label in df.columns):
            df[new_label] = df[label]
        else:
            df[new_label] += df[label]
    single_hero_labels = single_hero_labels2.copy()
    df = df.drop(labels=single_hero_labels,axis=1).copy()
    print("New Dataframe Colums:",df.columns,"\n")
    print("New Dataframe Shape:",df.shape,"\n")

    #print(df.query("d_firstblood_claimed == 0 and r_firstblood_claimed == 0").shape)

    for label in df.columns:
        if re.match(r"^(d|r)_\d*$",label):
            df = df.drop(label,axis=1)
    print(df.shape)

    target = target.loc[df.index]
    print(target.shape)
    df = df.drop('match_id_hash',axis=1)

    return df,target
