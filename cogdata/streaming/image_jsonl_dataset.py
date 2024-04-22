import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from copy import copy

class ImageJsonlDataset(Dataset):
    def __init__(self, jsonl_path, single_process_fn=None):
        """
        Initializes the ImageJsonlDataset.

        Args:
        jsonl_path (str): Path to the JSONL file where each line is a JSON object.
        single_process_fn (callable, optional): A callable function to process data items in __getitem__.
        """
        self.jsonl_path = jsonl_path
        self.single_process_fn = single_process_fn
        
        # Load JSONL file and store each line
        self.data = []
        with open(jsonl_path, 'r') as file:
            for line in file:
                try:
                    self.data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Failed to decode line: {line.strip()}")
    
    def __len__(self):
        """
        Returns the number of entries in the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset by index.

        Args:
        idx (int): Index of the data item in the dataset.

        Returns:
        Processed data item, which includes the image as a PIL object and other key-value pairs.
        """
        item = copy(self.data[idx])
        # Load the image
        image_path = item.get('imagepath')
        try:
            image = Image.open(image_path)
            item['image'] = image
        except IOError:
            print(f"Warning: Failed to open image at {image_path}. Returning None for 'image'.")
            item['image'] = None  # Indicate failure to load image
        
        # Apply the processing function if specified
        if self.single_process_fn:
            item = self.single_process_fn(item)
        
        return item
