# Caltech101 patch_size 8

# ViT transformer
# python vit_efficient_transformer.py \
# --model transformer \
# --dataset Caltech101 \
# --batch-size 32 \
# --test-batch-size 256 \
# > ./logs/Caltech101/vit_transformer_patch_size_8.log

# ViT performer

python vit_efficient_transformer.py \
--model softmax \
--nb_features 256 \
--redraw 1000 \
--dataset Caltech101 \
--batch-size 32 \
--test-batch-size 256 \
> ./logs/Caltech101/vit_performer_patch_size_8_softmax_nb_features_256_feature_redraw_interval_1000.log

python vit_efficient_transformer.py \
--model relu \
--nb_features 256 \
--redraw 1000 \
--dataset Caltech101 \
--batch-size 32 \
--test-batch-size 256 \
> ./logs/Caltech101/vit_performer_patch_size_8_relu_nb_features_256_feature_redraw_interval_1000.log

# python vit_efficient_transformer.py \
# --model softmax \
# --nb_features 64 \
# --redraw 1000 \
# --dataset Caltech101 \
# --batch-size 32 \
# --test-batch-size 256 \
# > ./logs/Caltech101/vit_performer_patch_size_8_softmax_nb_features_64_feature_redraw_interval_1000.log

# # ViT linformer
# python vit_efficient_transformer.py \
# --model linformer \
# --dataset Caltech101 \
# --batch-size 32 \
# --test-batch-size 256 \
# --k_linformer 64 \
# --one_kv_head \
# > ./logs/Caltech101/vit_linformer_patch_size_8_k_64_one_kv_head_True.log

# python vit_efficient_transformer.py \
# --model linformer \
# --dataset Caltech101 \
# --batch-size 32 \
# --test-batch-size 256 \
# --k_linformer 64 \
# > ./logs/Caltech101/vit_linformer_patch_size_8_k_64_one_kv_head_False.log

# # ViT reformer
# python vit_efficient_transformer.py \
# --model reformer \
# --dataset Caltech101 \
# --batch-size 32 \
# --n_bucket_size 4 \
# --n_hashes 2 \
# --test-batch-size 256 \
# > ./logs/Caltech101/vit_reformer_patch_size_8_n_bucket_size_4_n_hashes_2.log


# python vit_efficient_transformer.py \
# --model reformer \
# --dataset Caltech101 \
# --batch-size 32 \
# --n_bucket_size 16 \
# --n_hashes 2 \
# --test-batch-size 256 \
# > ./logs/Caltech101/vit_reformer_patch_size_8_n_bucket_size_16_n_hashes_2.log


# # ViT longformer
# python vit_efficient_transformer.py \
# --model longformer \
# --dataset Caltech101 \
# --batch-size 32 \
# --attention_window 256 \
# --test-batch-size 256 \
# > ./logs/Caltech101/vit_longformer_patch_size_8_attention_window_256.log

# python vit_efficient_transformer.py \
# --model longformer \
# --dataset Caltech101 \
# --batch-size 32 \
# --attention_window 128 \
# --test-batch-size 256 \
# > ./logs/Caltech101/vit_longformer_patch_size_8_attention_window_128.log


# # ViT Nystromformer
# python vit_efficient_transformer.py \
# --model nystromformer \
# --dataset Caltech101 \
# --batch-size 32 \
# --test-batch-size 256 \
# --num_landmarks 128 \
# --pinv_iterations 6 \
# > ./logs/Caltech101/vit_nystromformer_patch_size_8_num_landmarks_128_pinv_iterations_6.log

# python vit_efficient_transformer.py \
# --model nystromformer \
# --dataset Caltech101 \
# --batch-size 32 \
# --test-batch-size 256 \
# --num_landmarks 32 \
# --pinv_iterations 6 \
# > ./logs/Caltech101/vit_nystromformer_patch_size_8_num_landmarks_32_pinv_iterations_6.log




# Caltech256

# # ViT linformer
# python vit_efficient_transformer.py \
# --model linformer \
# --dataset Caltech256 \
# --batch-size 32 \
# --test-batch-size 256 \
# --k_linformer 64 \
# --one_kv_head \
# > ./logs/Caltech256/vit_linformer_patch_size_8_k_64_one_kv_head_True.log

# python vit_efficient_transformer.py \
# --model linformer \
# --dataset Caltech256 \
# --batch-size 32 \
# --test-batch-size 256 \
# --k_linformer 64 \
# > ./logs/Caltech256/vit_linformer_patch_size_8_k_64_one_kv_head_False.log

# # ViT reformer
# python vit_efficient_transformer.py \
# --model reformer \
# --dataset Caltech256 \
# --batch-size 32 \
# --n_bucket_size 4 \
# --n_hashes 2 \
# --test-batch-size 256 \
# > ./logs/Caltech256/vit_reformer_patch_size_8_n_bucket_size_4_n_hashes_2.log


# python vit_efficient_transformer.py \
# --model reformer \
# --dataset Caltech256 \
# --batch-size 32 \
# --n_bucket_size 16 \
# --n_hashes 2 \
# --test-batch-size 256 \
# > ./logs/Caltech256/vit_reformer_patch_size_8_n_bucket_size_16_n_hashes_2.log


# # ViT longformer
# python vit_efficient_transformer.py \
# --model longformer \
# --dataset Caltech256 \
# --batch-size 32 \
# --attention_window 256 \
# --test-batch-size 256 \
# > ./logs/Caltech256/vit_longformer_patch_size_8_attention_window_256.log

# python vit_efficient_transformer.py \
# --model longformer \
# --dataset Caltech256 \
# --batch-size 32 \
# --attention_window 128 \
# --test-batch-size 256 \
# > ./logs/Caltech256/vit_longformer_patch_size_8_attention_window_128.log


# # ViT Nystromformer
# python vit_efficient_transformer.py \
# --model nystromformer \
# --dataset Caltech256 \
# --batch-size 32 \
# --test-batch-size 256 \
# --num_landmarks 128 \
# --pinv_iterations 6 \
# > ./logs/Caltech256/vit_nystromformer_patch_size_8_num_landmarks_128_pinv_iterations_6.log

# python vit_efficient_transformer.py \
# --model nystromformer \
# --dataset Caltech256 \
# --batch-size 32 \
# --test-batch-size 256 \
# --num_landmarks 32 \
# --pinv_iterations 6 \
# > ./logs/Caltech256/vit_nystromformer_patch_size_8_num_landmarks_32_pinv_iterations_6.log

# Performer

python vit_efficient_transformer.py \
--model softmax \
--nb_features 256 \
--redraw 1000 \
--dataset Caltech256 \
--batch-size 32 \
--test-batch-size 256 \
> ./logs/Caltech256/vit_performer_patch_size_8_softmax_nb_features_256_feature_redraw_interval_1000.log

python vit_efficient_transformer.py \
--model relu \
--nb_features 256 \
--redraw 1000 \
--dataset Caltech256 \
--batch-size 32 \
--test-batch-size 256 \
> ./logs/Caltech256/vit_performer_patch_size_8_relu_nb_features_256_feature_redraw_interval_1000.log