import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 환경 변수 설정

model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
# model_id = "kyujinpy/Ko-PlatYi-6B"

# 토크나이저와 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# MPS 사용 가능 여부 확인
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.backends.cuda.is_built() else "cpu")
print("device:", device)

# 메모리 클리어
if device == "mps":
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    torch.mps.empty_cache()

# 모델을 MPS 장치로 이동
model.to(device)

messages = [
    {"role": "system", "content": "answer as if you're my best friend"},
    {"role": "user", "content": "안녕! 뭐하고 있어?"}
]

# 입력 데이터를 토큰화하고 텐서로 변환
encodes = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodes.to(device)


generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded)








# # 모델을 MPS 장치로 이동
# model.to(device)
#
# messages = [
#     {"role": "user", "content": "저녁 메뉴를 추천해줘"}
# ]
#
# # encodeds의 텐서 형식 확인
# encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
# print(encodeds)  # 반환값 구조 확인
#
# # 필요한 형식으로 변환
# if isinstance(encodeds, dict):
#     model_inputs = {k: v.to(device) for k, v in encodeds.items()}
#     input_ids = model_inputs["input_ids"]
#     attention_mask = model_inputs["attention_mask"]
# else:
#     model_inputs = encodeds.to(device)
#     input_ids = model_inputs
#     attention_mask = None
#
# generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=100, do_sample=True)
# decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
# print(decoded[0])
#
