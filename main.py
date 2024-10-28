import argparse
from train import load_and_prepare_data, create_dataset, setup_tokenizer, preprocess_example, train_model, predictions_to_map_output
from utils import read_prompt, generate_plan, write_result, use_cached_plan

prompt_file = './prompt/question_generation.txt'
result_file = './data/extra_data.csv'

def parse_args():
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description="Generate questions from LLM.")
    parser.add_argument('--query', action='store_true', help="Request quetions from LLM.")

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()

    if args.query:
        prompt_text = read_prompt(prompt_file)
    
        messages = [{
            'role': 'user',
            'content': [
                {
                    'text': prompt_text
                },
            ]
        }]
    
        print("Input Messages :")
        print(messages)
        print("-------------------------------------------------------")
        generated_text = generate_plan(messages)
    
        print("--------------------Generated Questions---------------------")
        print(generated_text)
        print("------------------------------------------------------------")
    
        write_result(result_file, generated_text)

'''
    # 加载训练数据
    train_df = load_and_prepare_data('./data/train.csv')
    train_ds = create_dataset(train_df)
    tokenizer = setup_tokenizer('./bert-base-cased')

    # 预处理训练集
    tokenized_train_ds = train_ds.map(preprocess_example, batched=False, remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])

    # 训练模型
    trainer = train_model(tokenized_train_ds)

    # 对测试集进行相同处理
    test_df = load_and_prepare_data('./data/test.csv')
    test_df['answer'] = 'A'  # 假设答案是A
    test_ds = create_dataset(test_df)
    tokenized_test_ds = test_ds.map(preprocess_example, batched=False, remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])

    # 预测
    test_predictions = trainer.predict(tokenized_test_ds)
    submission_df = test_df[['id']].copy()
    submission_df['prediction'] = predictions_to_map_output(test_predictions.predictions)
    submission_df.to_csv('submission.csv', index=False)
'''