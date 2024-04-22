import torch
from PIL import Image
from io import BytesIO
import torchvision
from torchvision import transforms

def _get_ltor_masks_and_position_ids(
    data: torch.Tensor,
    eod_token: int,
    reset_position_ids: bool,
    reset_attention_mask: bool,
    eod_mask_loss: bool,
):
    """Build masks and position id for left to right model.

    Args:
        data (torch.Tensor): The data tenor that holds the tokens from the dataset

        eod_token (int): ID of the token to that is considered the EOD

        reset_position_ids (bool): Switch to reset the document position ID's

        reset_attention_mask (bool): Switch to reset the attention mask

        eod_mask_loss (bool): Switch to enable the EOD mask loss

    Returns:
        torch.Tensor : Attention mask needed to be used for Attention

        torch.Tensor : The mask used for loss value during training

        torch.Tensor : The position ID's of the token
    """
    seq_length = data.numel()

    attention_mask = torch.tril(torch.ones((seq_length, seq_length), device=data.device)).unsqueeze(
        0
    )

    # Loss mask.
    loss_mask = torch.ones(seq_length, dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Find indices where EOD token is.
        eod_index = position_ids[data == eod_token]
        # Detach indices from positions if going to modify positions.
        if reset_position_ids:
            eod_index = eod_index.clone()

        # Loop through EOD indices:
        prev_index = 0
        for j in range(eod_index.numel()):
            i = eod_index[j]
            # Mask attention loss.
            if reset_attention_mask:
                attention_mask[0, (i + 1) :, : (i + 1)] = 0
            # Reset positions.
            if reset_position_ids:
                position_ids[(i + 1) :] -= i + 1 - prev_index
                prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids


class ImageTextPairProcessFnBuilder:
    def __init__(self, img_size, max_text_len, tokenizer_name):
        self.img_size = img_size
        self.max_text_len = max_text_len
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
        local_files_only=True
    )

    def __call__(self, src):
        for data in src:
            img_bytes = data['png'] if 'png' in data else data['jpg']
            try:
                img = Image.open(BytesIO(img_bytes)).convert('RGB')
            except Exception as e:
                print(e)
                continue
            img = img.resize((self.img_size,self.img_size))
            # img = np.asarray(img)
            img = torchvision.transforms.functional.to_tensor(img)
            text = data['txt']
            if isinstance(text, bytes):
                text = text.decode('utf-8')
            text = self.tokenizer.encode(text)
            text = text[:self.max_text_len]
            text = torch.tensor(text)
            caption = data.get('caption', None)
            caption_zh = data.get('caption_zh', None)
            yield {
                'img': img,
                'txt': text,
                'caption': caption,
                'caption_zh': caption_zh,
                '__url__': data['__url__'],
                '__key__': data['__key__'],
            }

class ImageJsonlProcessFnBuilder:
    def __init__(self, img_size):
        """
        Initializes the ImageJsonlProcessFnBuilder with a target image size.

        Args:
        img_size (tuple): A tuple (width, height) specifying the target size for image resizing.
        """
        self.img_size = img_size
        self.transforms = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor() 
        ])

    def __call__(self, item):
        """
        Processes an item from ImageJsonlDataset to resize the image and convert it to PyTorch tensor format.

        Args:
        item (dict): A dictionary containing the 'image' key with a PIL Image object and other data.

        Returns:
        dict: The dictionary with the processed image and other original key-value pairs.
        """
        if item['image'] is not None:
            # Apply the transformations to the image
            item['image'] = self.transforms(item['image'])
        return item


class TextJonslProcessFnBuilder:
    def __init__(self, max_sequence_length, tokenizer_name):
        from transformers import AutoTokenizer
        self.max_text_len = max_sequence_length + 1
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
            local_files_only=True
        )
    def __call__(self, src):
        buffered_token_ids = None
        for x in src:
            txt = x['text']
            tokenized_ids = torch.tensor(self.tokenizer.encode(txt), dtype=torch.long)
            if buffered_token_ids is None:
                buffered_token_ids = tokenized_ids
            else:
                buffered_token_ids = torch.cat((buffered_token_ids, tokenized_ids), dim=0)
            # yield per seq_len
            this_key_times = 0
            while buffered_token_ids.shape[0] >= self.max_text_len:
                text = buffered_token_ids[:self.max_text_len]
                buffered_token_ids = buffered_token_ids[self.max_text_len:]

                labels = text[1:].contiguous()
                tokens = text[:-1].contiguous()

                # assert not torch.any(
                #     tokens >= self.vocab_size
                # ), "An input token is out of bounds of the tokenizer vocabulary"

                attention_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
                    tokens,
                    self.tokenizer.eos_token_id,
                    reset_position_ids=False,
                    reset_attention_mask=False,
                    eod_mask_loss=False,
                )

                yield {
                    "tokens": tokens,
                    "labels": labels,
                    "attention_mask": attention_mask,
                    "loss_mask": loss_mask,
                    "position_ids": position_ids,
                    '__url__': x['__url__'],
                    '__key__': x['__key__'] + f'@{this_key_times}',
                }
                this_key_times += 1