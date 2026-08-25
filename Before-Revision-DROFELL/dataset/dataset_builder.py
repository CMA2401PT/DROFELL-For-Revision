from dataclasses import dataclass
import os
import json
import pickle
from hashlib import md5
from typing import Any, Callable, Iterable, List, Optional, Dict
import torch
from torch.utils.data import Dataset
from .common import ProcessedRecord, Record


@dataclass
class DataSetSampleBatchMeta:
    position_ids_tensor: torch.Tensor
    samples_input_ids: torch.Tensor
    samples_label_ids: torch.Tensor
    input_lens: List[int]


class SimpleDataset(Dataset[ProcessedRecord]):
    def __init__(self, records: List[ProcessedRecord]) -> None:
        super().__init__()
        self.records: List[ProcessedRecord] = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ProcessedRecord:
        return self.records[index]


class DataCollatorFlatten:
    def __init__(self):
        pass

    def __call__(self, samples: List[ProcessedRecord]):
        input_lens = [len(sample.input_ids) for sample in samples]
        position_ids_tensor: torch.Tensor = torch.tensor(data=[i for input_len in input_lens for i in range(input_len)], dtype=torch.int32)
        input_ids_tensor = torch.tensor([e for sample in samples for e in sample.input_ids], dtype=torch.int32)
        label_ids_tensor = torch.tensor([e for sample in samples for e in sample.output_ids], dtype=torch.long)
        return DataSetSampleBatchMeta(position_ids_tensor=position_ids_tensor, samples_input_ids=input_ids_tensor, samples_label_ids=label_ids_tensor, input_lens=input_lens)


def build_dataset(source_name: str, source_provide_fn: Callable[[], Iterable[Record]],
                  process_fn_name: str, process_fn: Callable[..., Callable[[Record], ProcessedRecord]],
                  process_fn_args: Optional[List[str]], process_fn_kwargs: Optional[Dict[str, Any]],
                  cache_dir: str,
                  non_hashable_process_fn_kwargs: Optional[Dict[str, Any]] = None,
                  ) -> SimpleDataset:
    os.makedirs(cache_dir, exist_ok=True)
    # source_cache = os.path.join(cache_dir, md5(source_name.encode("utf-8")).hexdigest() + ".pkl")
    process_fn_full_description = process_fn_name
    if process_fn_args is not None:
        process_fn_full_description += ",args=" + json.dumps(process_fn_args)
    if process_fn_kwargs is not None:
        sorted_kw_args = list(sorted(process_fn_kwargs.items()))
        process_fn_full_description += ",kwargs=" + json.dumps(sorted_kw_args)
    if process_fn_kwargs is None:
        process_fn_kwargs = {}
        if non_hashable_process_fn_kwargs is not None:
            process_fn_kwargs.update(non_hashable_process_fn_kwargs)
    full_description = source_name + "-> [" + process_fn_full_description + "]"
    full_description_md5 = md5(full_description.encode("utf-8")).hexdigest()
    final_cache = os.path.join(cache_dir, full_description_md5 + ".pkl")
    processed_records: List[ProcessedRecord] = []
    if os.path.exists(final_cache):
        print("reading processed records...")
        with open(final_cache, "rb") as f:
            processed_records = pickle.load(f)
        print("processed dataset loaded")
    else:
        print("processing dataset...")
        if process_fn_args is None:
            process_fn_args = []
        post_processer = process_fn(*process_fn_args, **process_fn_kwargs)
        processed_records = []
        for record in source_provide_fn():
            processed_records.append(post_processer(record))
        with open(final_cache, "wb") as f:
            pickle.dump(processed_records, f)
        print("dataset generated and cached")
    return SimpleDataset(processed_records)
