import argparse
from train import load_and_prepare_data, create_dataset, setup_tokenizer, preprocess_example, train_model, predictions_to_map_output
from utils import read_prompt, generate_plan, write_result, use_cached_plan


result_dir = './data/extra_data.csv'

def parse_args():
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser(description="No.")
    parser.add_argument('--query', action='store_true', help="Request quetions from LLM.")
    parser.add_argument('--prompt_dir', default='./prompt/question_generation.txt', help='Directory containing prompt files')
    parser.add_argument('--train_dir', default='./data/train_data_6k.csv' , help='Directory containing the input CSV training file')
    parser.add_argument('--test_dir', default='./data/test_data_0.5k.csv', help='Directory containing the input CSV training file')
    parser.add_argument('--model_dir', default='./bert-base-cased', help='Directory containing pre-trained model files')
    parser.add_argument('--save_dir', default='./model/finetuned_bert', help='Directory to save your model files')

    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    
    is_query = args.query
    prompt_dir = args.prompt_dir
    train_dir = args.train_dir
    test_dir = args.test_dir
    model_dir = args.model_dir
    save_dir = args.save_dir

    if is_query:
        prompt_text = read_prompt(prompt_dir)
    
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
    
        write_result(result_dir, generated_text)

    # 加载训练数据
    if args.query:
        train_dir = result_dir
    train_df = load_and_prepare_data(train_dir)
    train_ds = create_dataset(train_df)
    tokenizer = setup_tokenizer(model_dir)
  
    # 预处理训练集
    tokenized_train_ds = train_ds.map(preprocess_example, batched=False, remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])

    # 训练模型
    trainer = train_model(tokenized_train_ds)

    # 对测试集进行相同处理
    test_df = load_and_prepare_data(test_dir)
    test_df['answer'] = 'A'  # 假设答案是A
    test_ds = create_dataset(test_df)
    tokenized_test_ds = test_ds.map(preprocess_example, batched=False, remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])

    # 预测
    test_predictions = trainer.predict(tokenized_test_ds)
    submission_df = test_df[['id']].copy()
    submission_df['prediction'] = predictions_to_map_output(test_predictions.predictions)
    submission_df.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()