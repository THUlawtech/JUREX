python step0-generate_embedding.py \
    --encoder BAAI/bge-m3 \
    --candidates /root/bge/Lecard/GEAR_candidate/candidate_gpt_GEAR_llm_qwen25-72b \
    --languages ar de en es fr hi it ja ko pt ru th zh \
    --index_save_dir ./corpus-index \
    --max_passage_length 8192 \
    --batch_size 4 \
    --fp16 \
    --pooling_method cls \
    --normalize_embeddings True


#    encoder baselines:
#    fengshan/ChatLaw-Text2Vec
#    /root/autodl-tmp/Xorbits/bge-m3
#    /SAILER
