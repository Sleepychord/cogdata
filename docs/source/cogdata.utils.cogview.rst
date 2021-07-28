cogdata.utils.cogview package
=============================

cogdata.utils.cogview.api module
--------------------------------

.. automodule:: cogdata.utils.cogview.api
   :members:
   :undoc-members:
   :show-inheritance:

cogdata.utils.cogview.sp\_tokenizer module
------------------------------------------

SentencePiece tokenizer. from https://github.com/openai/gpt-2/, changed for chinese

SentencePiece is an unsupervised text tokenizer and detokenizer mainly for Neural Network-based text generation 
systems where the vocabulary size is predetermined prior to the neural model training. SentencePiece implements 
subword units (e.g., byte-pair-encoding (BPE) [Sennrich et al.]) and unigram language model [Kudo.]) with the 
extension of direct training from raw sentences. SentencePiece allows us to make a purely end-to-end 
system that does not depend on language-specific pre/postprocessing.
https://github.com/google/sentencepiece

::

   pip install sentencepiece

or::

   git clone https://github.com/google/sentencepiece.git
   python setup.py install

cogdata.utils.cogview.unified\_tokenizer module
-----------------------------------------------
.. automodule:: cogdata.utils.cogview.unified_tokenizer
   :special-members: get_tokenizer


cogdata.utils.cogview.vqvae\_tokenizer module
---------------------------------------------
This module defines the tokenizer used in *Cogdata*


cogdata.utils.cogview.vqvae\_zc module
--------------------------------------
This module defines the model used in *Cogdata*

