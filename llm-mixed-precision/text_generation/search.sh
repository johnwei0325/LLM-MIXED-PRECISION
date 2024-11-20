# llama
    
## mpl    
    
for SIZE in  gpt2 # llama-7b # llama-13b # llama-30b
    do
    for BIT in 8 6
        do
python search.py \
    --model $SIZE --target_model GPT2LMHeadModel \
    --job_dir search_experiment/${SIZE}/mpl_85_m_1/t_${BIT}_0/ \
    --method wikitext --max_seq_len 1024 \
    --qmethod mpl \
    --bitW $BIT --abitW $BIT \
    --source_file ./search_experiment/${SIZE}/mpl_85_m_1/t_${BIT}_0/ \
    --finetuned True \
    --bit_search_only False \
    --test_only False
    --port 29518 \
    --gpus -1 \
    --num_epochs 50
    --train_batch_size 8
    --eval_batch_size 2
    
        done
    done