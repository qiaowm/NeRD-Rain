python test.py\
    --input_dir './Datasets/real-ds/input' \
    --output_dir './Datasets/real-ds/result_pretrain' \
    --weights './checkpoints/model_large_SPA.pth' \
    --win_size 128
python test.py\
    --input_dir './Datasets/real-ds/input' \
    --output_dir './Datasets/real-ds/result_ft' \
    --weights './checkpoints/model_best_ds.pth' \
    --win_size 128