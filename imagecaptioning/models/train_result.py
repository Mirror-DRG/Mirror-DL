from transformers import pipeline
# 내 모델
image_to_text = pipeline("image-to-text", model="./image-captioning-test-output")
# 이미 있는거 갖다 쓰는 코드
#image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

generate_kwargs = {
   "num_return_sequences":3,
    "num_beams":3
}
result = image_to_text("test2.jpg", generate_kwargs=generate_kwargs)
#result = image_to_text("https://ankur3107.github.io/assets/images/image-captioning-example.png", generate_kwargs=generate_kwargs)

print(result)