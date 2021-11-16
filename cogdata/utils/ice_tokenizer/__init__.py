from .sp_tokenizer import from_pretrained

def get_tokenizer(img_tokenizer_path=None):
    """Singlton
    Return an image tokenizer"""
    if not hasattr(get_tokenizer, 'tokenizer'):
        # the first time to load the tokenizer, specify img_tokenizer_path
        get_tokenizer.tokenizer = from_pretrained()
    return get_tokenizer.tokenizer