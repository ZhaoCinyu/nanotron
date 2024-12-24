python tools/preprocess_data.py \
       --tokenizer-name-or-path HuggingFaceTB/cosmo2-tokenizer \
       --output-folder datasets/smollm_corpus_tiny \
       --n-tasks 16 \
       hf \
       --dataset rellabear/smollm_corpus_subset \