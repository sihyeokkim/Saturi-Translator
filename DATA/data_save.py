import pandas as pd
import datasets
from tqdm import tqdm

def data_save(data,pipe,reg,num_rows, start,directory) :
    dataset = data.loc[data['reg'] == reg]
    for i in tqdm(range(start, len(data), num_rows)) :
        dataset = dataset[i:i+num_rows]
        dataset = datasets.Dataset.from_pandas(dataset)
        dataset = dataset.map(lambda ds : {'eng' : pipe(ds['text'])[0]['translation_text']})
        dataset.set_format('pandas')
        dataset[:].to_csv(directory + f'dataset_full_en_kor_{reg}_v{i}_{i+num_rows}.csv')
