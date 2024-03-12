# data 관련
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd

# model 관련
import torch
import numpy as np
import evaluate
import nltk
from transformers import VisionEncoderDecoderConfig, VisionEncoderDecoderModel, ViTConfig,\
    BertConfig, BertTokenizer, ViTImageProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments,\
    default_data_collator, pipeline

import os
import nltk

os.environ["WANDB_DISABLED"] = "true"

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", quiet=True)

# mac gpu 설정
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# window & linux gpu 설정
device = "cuda:0" if torch.cuda.is_available() else "cpu"

image_encoder_model = "google/vit-base-patch16-224-in21k"
text_decode_model = "bert-base-uncased"

model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(image_encoder_model, text_decode_model).to(device)
feature_extractor = ViTImageProcessor.from_pretrained(image_encoder_model)
tokenizer = BertTokenizer.from_pretrained(text_decode_model)

# model config 설정
#config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(ViTConfig(), BertConfig())
#model.config = config
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.bos_token_id = tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.sep_token_id = tokenizer.sep_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.max_length = 70
model.config.decoder.max_length = 70

# model config 및
# output_dir = "vit-bert-image-captioning"
# model.save_pretrained(output_dir)
# feature_extractor.save_pretrained(output_dir)
# tokenizer.save_pretrained(output_dir)

# data 불러오기
# data 경로 설정
data_dir = '../../rawdataset/coco'
data_type = 'train2017'
val_data_type = 'val2017'

# csv 파일 경로
train_csv = '../preprocessing/pro_cap_small_{}.csv'.format(data_type)
train_data = pd.read_csv(train_csv)
val_csv = '../preprocessing/pro_cap_{}.csv'.format(val_data_type)
val_data = pd.read_csv(val_csv)

# coco dataset 전처리 클래스
class ImageCaptioningDataset(Dataset):
    def __init__(self, data, data_dir, data_type, max_target_length):
        self.data = data
        self.data_dir = data_dir
        self.data_type = data_type
        self.max_target_length = max_target_length

    def __getitem__(self, idx):
        print(idx)
        image_path = '{}/{}/{}'.format(self.data_dir, self.data_type, self.data.loc[idx]["file_name"])
        caption = self.data.loc[idx]["caption"]

        model_inputs = dict()
        model_inputs['labels'] = self.tokenization_fn(caption, self.max_target_length)
        model_inputs['pixel_values'] = self.feature_extraction_fn(image_path)

        return model_inputs

    def __len__(self):
        return len(self.data)

    # caption 전처리 함수
    def tokenization_fn(self, caption, max_target_length):
        labels = tokenizer(caption, padding="max_length", max_length=max_target_length).input_ids

        return labels

    # 이미지 전처리 함수
    def feature_extraction_fn(self, image_path):
        image = Image.open(image_path).convert("RGB")
        encoder_inputs = feature_extractor(images=image, return_tensor="pt")
        pixel_values_tensor = torch.tensor(encoder_inputs["pixel_values"][0])
        # pixel_values_tensor = pixel_values_tensor.unsqueeze(0)

        return pixel_values_tensor


# 검증 척도 설정
metric = evaluate.load("rouge")
ignore_pad_token_for_loss = True

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    if ignore_pad_token_for_loss:
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds,
                            references=decoded_labels,
                            use_stemmer=True)

    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)

    return result


# 모델 학습
train_ds = ImageCaptioningDataset(train_data, data_dir, data_type, 70)
val_ds = ImageCaptioningDataset(val_data, data_dir, val_data_type, 70)

# 학습 인자 정의 및 모델 학습
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    save_strategy="steps",
    save_steps=50000,
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    output_dir="image-captioning-output",
    fp16=True,
    fp16_opt_level='03',
)
trainer = Seq2SeqTrainer(
    model = model,
    tokenizer = feature_extractor,
    args = training_args,
    compute_metrics = compute_metrics,
    train_dataset = train_ds,
    eval_dataset = val_ds,
    data_collator = default_data_collator,
)

trainer.train()
trainer.save_model("./image-captioning-output")
tokenizer.save_pretrained("./image-captioning-output")