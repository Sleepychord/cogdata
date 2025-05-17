# -*- encoding: utf-8 -*-
"""
ParquetDataset for reading parquet files.
"""

from torch.utils.data import Dataset

from cogdata.utils.register import register

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError as e:  # pragma: no cover - pyarrow optional
    pq = None


@register
class ParquetDataset(Dataset):
    """Dataset for Apache Parquet files."""

    def __init__(self, path, columns=None, transform_fn=None):
        """Load a parquet file.

        Parameters
        ----------
        path : str
            Path to the parquet file.
        columns : list of str, optional
            Columns to load. ``None`` means load all columns.
        transform_fn : callable, optional
            Function applied to each record in ``__getitem__``.
        """
        if pq is None:
            raise ModuleNotFoundError(
                "pyarrow is required to use ParquetDataset")

        self.path = path
        self.columns = columns
        self.transform_fn = transform_fn
        self.table = pq.read_table(path, columns=columns)
        self.length = self.table.num_rows

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        row_table = self.table.slice(index, 1)
        record = {k: v[0] for k, v in row_table.to_pydict().items()}
        if self.transform_fn is not None:
            record = self.transform_fn(record)
        return record
