import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from datasets import Dataset
from dataclasses import dataclass
from typing import Optional, Union

model_dir = './bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(model_dir)

def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """加载CSV数据，并进行必要的类型转换"""
    df = pd.read_csv(file_path)
    for col in ['prompt', 'A', 'B', 'C', 'D', 'E']:
        df[col] = df[col].astype(str)
    return df

def create_dataset(df: pd.DataFrame) -> Dataset:
    """从pandas DataFrame创建HuggingFace的Dataset对象"""
    return Dataset.from_pandas(df)

def setup_tokenizer(model_dir: str) -> AutoTokenizer:
    """初始化tokenizer"""
    return AutoTokenizer.from_pretrained(model_dir)

def preprocess_example(example, options='ABCDE'):
    """对单个样本进行预处理，以适应多选模型输入格式"""
    
    first_sentence = [example['prompt']] * len(options)
    second_sentence = [example[option] for option in options]
    tokenized_example = tokenizer(first_sentence, second_sentence, truncation=True)
    tokenized_example['label'] = {option: index for index, option in enumerate(options)}[example['answer']]
    return tokenized_example

@dataclass
class DataCollatorForMultipleChoice:
    """自定义的数据收集器，用于多选项选择任务"""
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if 'label' in features[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch

def train_model(tokenized_train_ds, model_dir='./bert-base-cased', save_dir='./model/finetuned_bert'):
    """训练模型并返回Trainer实例"""
    model = AutoModelForMultipleChoice.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    training_args = TrainingArguments(
        output_dir=model_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        report_to='none'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_train_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    )

    trainer.train()
    trainer.save_model(save_dir)
    print(f"Model saved at :{save_dir}.")
    return trainer

def predictions_to_map_output(predictions, options='ABCDE'):
    """将预测结果转换成最终提交格式"""
    sorted_answer_indices = np.argsort(-predictions, axis=-1)
    top_answer_indices = sorted_answer_indices[:, :3]
    top_answers = np.vectorize({index: option for index, option in enumerate(options)}.get)(top_answer_indices)
    return np.apply_along_axis(lambda row: ' '.join(row), 1, top_answers)
