from torch.utils.data import Dataset

from xturing.registry import BaseParent


class BaseDataset(BaseParent, Dataset):
    @property
    def meta(self):
        return self._meta
