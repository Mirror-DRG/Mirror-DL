from transformers import pipeline
# 내 모델
#image_to_text = pipeline("image-to-text", model="./image-captioning-output")
# 업로드한 모델 갖고오기
image_to_text = pipeline("image-to-text", model="roomie00/vit-bert-image-captioning")

generate_kwargs = {
   "num_return_sequences":3,
    "num_beams":5,
    "max_length":50
}
result = image_to_text("test3.jpg", generate_kwargs=generate_kwargs)
#result = image_to_text("https://ankur3107.github.io/assets/images/image-captioning-example.png", generate_kwargs=generate_kwargs)

print(result)