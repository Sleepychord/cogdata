from .zip_dataset import ZipDataset
from .tar_dataset import TarDataset
from .binary_dataset import BinaryDataset
from ..utils.logger import get_logger
# try:
#     from .rar_dataset import StreamingRarDataset
# except LookupError:
#     get_logger().warning("Couldn't find path to unrar library, skipping.\n StreamingRarDataset cannot be imported now. Run install_unrarlib.sh.")
