CUDA_VISIBLE_DEVICES=3,4 python3 code/survey.py \
  --model_path "[MODEL PATH] e.g. gpt-4o-mini-2024-07-18" \
  --prompt_mode "[CHOOSE MODE] default or korea" \
  --dataset_path dataset/KoValPlus.json \
  --key "[INSERT YOUR KEY] hf_token or OpenAI key"