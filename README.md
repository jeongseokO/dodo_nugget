# Dodo: Dynamic Contextual Compression for Decoder-only LMs

To set up environment, install anaconda3 and run
```shell
conda create -n dodo python=3.10
conda activate dodo
pip install -r requirements.txt
```

Make sure you have access to the 'meta-llama/Llama-2-7b-chat-hf' model at Huggingface. Then the `inference.py` file provides an example on the inference with an example in SQuAD.
It compresses the contet of SQuAD into nuggets and decodes the answer from context and question.
