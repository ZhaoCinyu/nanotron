import os
from datasets import load_dataset
from datasets import Dataset, DatasetDict

os.environ["HF_HOME"] = "/playpen/xinyu"

# dataset = load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", split="train", streaming=True)
# shuffled_dataset = dataset.shuffle(seed=42, buffer_size=10_000)
# dataset_head = shuffled_dataset.take(307137)
# ds = Dataset.from_generator(lambda: (yield from dataset_head), features=dataset_head.features)
# ds.save_to_disk("/playpen/xinyu/smollm/python-edu")

# dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
# shuffled_dataset = dataset.shuffle(seed=42, buffer_size=10_000)
# dataset_head = shuffled_dataset.take(1565360)
# ds = Dataset.from_generator(lambda: (yield from dataset_head), features=dataset_head.features)
# ds.save_to_disk("/playpen/xinyu/smollm/cosmopedia-v2")

dataset = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", split="train", streaming=True)
shuffled_dataset = dataset.shuffle(seed=42, buffer_size=10_000)
dataset_head = shuffled_dataset.take(7606720)
ds = Dataset.from_generator(lambda: (yield from dataset_head), features=dataset_head.features)
ds.save_to_disk("/playpen/xinyu/smollm/fineweb-edu-dedup")


# def add_id_and_filepath(example, idx):
#     example['id'] = f"cosmopedia-v2-{idx}"
#     example['file_path'] = example['id']
#     return example

# ds1 = ds1.map(add_id_and_filepath, with_indices=True)

# def move_metadata_columns(example):
#     for key in list(example['metadata'].keys()):
#         if key == 'file_path':
#             example[key] = example['metadata'][key]
#             # del example['metadata'][key]
#     del example['metadata']
#     return example

# ds2 = ds2.map(move_metadata_columns)

# from datasets import concatenate_datasets

# # ds1.features
# # First prepare ds2 by adding missing columns from ds1 with null values
# columns_to_add = [col for col in ds1.features if col not in ds2.features]

# # Add columns one by one using add_column
# ds2_with_nulls = ds2
# for col in columns_to_add:
# 	ds2_with_nulls = ds2_with_nulls.add_column(col, [None] * len(ds2))

# # Now concatenate the datasets
# ds1 = concatenate_datasets([ds1, ds2_with_nulls])

# ds1.save_to_disk("/playpen/xinyu/smollm/combined")