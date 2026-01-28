import csv
import math
import numpy as np
from collections import defaultdict
import pandas as pd

def process_csv(input_file, train_file, val_file, test_file, target_length=8):
    user_sequences = defaultdict(list)
    item_to_int = {}
    item_id_counter = 1
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            user_id, item_id, rating, timestamp = row
            if  int(float(rating)) >= 4:
                user_sequences[user_id].append((item_id, int(timestamp)))

    formatted_sequences = {}
    for user_id, interactions in user_sequences.items():
        sorted_interactions = sorted(interactions, key=lambda x: x[1])  
        items = [item for item, _ in sorted_interactions]
        if len(items) >= 5:  
            formatted_sequences[user_id] = items

    processed_data = []
    for user_id, seq in formatted_sequences.items():
        if len(seq) > target_length+1:
            seq = seq[-(target_length+1):] 
        
        for item_id in seq:
            if item_id not in item_to_int:
                item_to_int[item_id] = item_id_counter
                item_id_counter += 1
        seq = [item_to_int[item_id] for item_id in seq]
    
        target_item = seq[-1]  
        input_seq = seq[:-1]  
        seq_length = len(input_seq)
         
        if seq_length < target_length:
            input_seq = [0] * (target_length - seq_length)  + input_seq  

        processed_data.append((input_seq, seq_length, target_item)) 
    print(len(item_to_int))

    np.random.shuffle(processed_data)
    train_size = math.floor(len(processed_data) * 0.8)
    val_size = math.floor(len(processed_data) * 0.1)
    
    train_data = processed_data[:train_size]
    val_data = processed_data[train_size:train_size + val_size]
    test_data = processed_data[train_size + val_size:]

    def save_to_file(data, file_path):
        df = pd.DataFrame(data, columns=['seq', 'len_seq', 'next'])
        df.to_pickle(file_path)

    save_to_file(train_data, train_file)
    save_to_file(val_data, val_file)
    save_to_file(test_data, test_file)

input_csv = "./dataset/Amazon/ratings_Movies_and_TV.csv"
train_df = "./data/movie/train.df"
val_df = "./data/movie/val.df"
test_df = "./data/movie/test.df"

process_csv(input_csv, train_df, val_df, test_df, target_length=8)

