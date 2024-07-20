from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "kyujinpy/Ko-PlatYi-6B"
cache_dir = "./model_cache"

# 모델과 토크나이저를 로컬 캐시 디렉토리에 저장
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir)

# 로컬에 저장된 모델과 토크나이저를 사용할 수 있도록 설정
tokenizer.save_pretrained(cache_dir)
model.save_pretrained(cache_dir)
