from typing import List, Tuple
import datasets
from datasets import load_dataset, concatenate_datasets 

class DatasetProcesser(object):
    def __init__(self, dataset_names: List[str]):
        self.dataset_names = dataset_names

    def get_processed_train_eval_split(self, train_percent: float) -> Tuple[datasets.Dataset]:
        datasets = []
        for dataset_name in self.dataset_names:
            dataset = self.__load_dataset(dataset_name)
            dataset = self.__filter_english(dataset)
            dataset = self.__rename_columns(dataset)
            datasets.append(dataset) 

        concatenated_dataset = concatenate_datasets(datasets)
        test_dataset = concatenated_dataset.remove_columns(['input', 'output'])
        test_dataset = test_dataset.rename_column('instruction', 'text')
        concatenated_dataset = concatenated_dataset.map(self.__create_alpaca_prompt)
        concatenated_dataset = concatenated_dataset.remove_columns(['instruction', 'input', 'output'])


        assert 0 <= train_percent <= 1, "Split (train percent) must be between 0 and 1"
        dataset_dict = concatenated_dataset.train_test_split(train_size=train_percent, shuffle=True)

        return (dataset_dict['train'], dataset_dict['test'], test_dataset)

    def __load_dataset(self, dataset_name: str) -> datasets.Dataset:
        return load_dataset(dataset_name, split='train')
    
    def __filter_english(self, dataset: datasets.Dataset) -> datasets.Dataset:
        column_names = dataset.column_names
        en_columns = [col for col in column_names if col.startswith('en-')]
        return dataset.remove_columns(en_columns)
    
    def __rename_columns(self, dataset: datasets.Dataset) -> datasets.Dataset:
        column_names = dataset.column_names
        column_mapping = {s: s.split('-', 1)[-1] for s in column_names}
        return dataset.rename_columns(column_mapping)
    

    def __create_alpaca_prompt(self, row):
        def prompt_input(row):
            return ("Below is an instruction that describes a task, paired with an input that provides further context. "
                    "Write a response that appropriately completes the request.\n\n"
                    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}").format_map(row)
        
        def prompt_no_input(row):
            return ("Below is an instruction that describes a task. "
                    "Write a response that appropriately completes the request.\n\n"
                    "### Instruction:\n{instruction}\n\n### Response:\n{output}").format_map(row)
        
        row['text'] = prompt_no_input(row) if row["input"] == "" else prompt_input(row)
        return row
