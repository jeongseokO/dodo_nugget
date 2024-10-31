from typing import *
import os

from torch.utils import data
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import PreTrainedTokenizer, AutoTokenizer, LlamaTokenizer


class BaseLazyDataset:
    def __init__(
            self, tokenizer_name: str, tokenizer_kwargs: Dict[str, Any] = None,
            data_name: Optional[str] = None, data_args: Optional[Iterable] = None, split: Optional[str] = None,
    ):
        self._tokenizer_name, self._tokenizer_kwargs = tokenizer_name, tokenizer_kwargs
        self._data_name, self._data_args, self._split = data_name, data_args, split
        self._data: Optional[DatasetDict] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None

    def init(self):
        if self.tokenizer is None:
            kwargs = dict() if self._tokenizer_kwargs is None else self._tokenizer_kwargs
            if 'llama' in self._tokenizer_name.lower():
                self.tokenizer = LlamaTokenizer.from_pretrained(self._tokenizer_name, legacy=False, **kwargs)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name, **kwargs)
        if self._data is None:
            self._prepare_data()

    def clean(self):
        self.tokenizer = self._data = None

    def _prepare_data(self):
        self._data = self._get_data()

    def _get_data(self):
        if self._data_name is None:
            return
        if os.path.exists(self._data_name):
            path = self._data_name
            if self._split is not None:
                path = os.path.join(path, self._split)
            ret = load_from_disk(path)
        else:
            ret = load_dataset(self._data_name, *self._data_args, split=self._split)
        return ret


class IterableLazyDataset(BaseLazyDataset, data.IterableDataset):
    pass


class LazyDataset(BaseLazyDataset, data.Dataset):
    def __init__(
            self, tokenizer_name: str, tokenizer_kwargs: Dict[str, Any] = None,
            data_name: Optional[str] = None, data_args: Optional[Iterable] = None,
            split: Optional[str] = None, data_length: Optional[int] = None
    ):
        data_args = data_args or ()
        super().__init__(tokenizer_name, tokenizer_kwargs, data_name, data_args, split)
        tmp_data = self._get_data()
        if tmp_data is not None:
            real_data_length = len(tmp_data)
            if data_length is None:
                self._data_length = real_data_length
            else:
                self._data_length = min(real_data_length, data_length)

    def __len__(self):
        return self._data_length

    def clean(self):
        self.tokenizer = self._data = None
