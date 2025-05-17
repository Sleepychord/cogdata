import pytest

pyarrow = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq
import pyarrow as pa

from cogdata.datasets import ParquetDataset


def test_parquet_dataset(tmp_path):
    table = pa.table({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
    file_path = tmp_path / 'test.parquet'
    pq.write_table(table, file_path)

    ds = ParquetDataset(str(file_path))
    assert len(ds) == 3
    assert ds[1]['a'] == 2
    assert ds[2]['b'] == 'z'
