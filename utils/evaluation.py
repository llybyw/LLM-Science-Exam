import pandas as pd
from collections import defaultdict
def calculate_map_at_k(actual, predicted, k=3):
    """
    Calculate the Mean Average Precision at k (MAP@k).
    
    :param actual: List of actual answers for a single question.
    :param predicted: List of predicted answers for a single question.
    :param k: Cutoff value for precision.
    :return: MAP@k for the given question.
    """
    if not actual:
        return 0.0
    
    correct = set(actual)
    num_correct = len(correct)
    precision_sum = 0.0
    relevant_count = 0
    
    for i, pred in enumerate(predicted[:k]):
        if pred in correct:
            relevant_count += 1
            precision_sum += relevant_count / (i + 1)
            correct.remove(pred)
            if not correct:
                break
    print(precision_sum)
    return precision_sum / min(num_correct, k)

def evaluate_map_at_3(test_file, prediction_file):
    """
    Evaluate the Mean Average Precision at 3 (MAP@3) for the test set and predictions.
    
    :param test_file: Path to the test CSV file.
    :param prediction_file: Path to the prediction CSV file.
    """
    # Read the test and prediction files
    test_df = pd.read_csv(test_file)
    pred_df = pd.read_csv(prediction_file)

    # Create a dictionary to store the predictions
    predictions = defaultdict(str)
    for _, row in pred_df.iterrows():
        # Store the prediction string
        predictions[row['id']] = row['prediction']

    # Initialize the sum of MAP@3 scores
    map_at_3_sum = 0.0
    num_questions = len(test_df)

    # Calculate MAP@3 for each question and print the results
    for _, row in test_df.iterrows():

        question_id = row['id']
        actual_answer = row['answer']
        
        # Get the prediction string and split it into a list of characters
        predicted_string = predictions[question_id]
        predicted_answers = predicted_string.strip("[]").split()

        

        # Print the correct answer and the top 3 predicted answers
        print(f"Question ID: {question_id}")
        print(f"Correct Answer: {actual_answer}")
        print(f"Predicted Answers: {predicted_answers}")

        # Debugging: Check if the actual answer is in the predicted answers
        if actual_answer in predicted_answers:
            print(f"Correct answer {actual_answer} found in predictions.")

        else:
            print(f"Correct answer {actual_answer} NOT found in predictions.")

        # Calculate MAP@3 for this question
        map_at_3 = calculate_map_at_k([actual_answer], predicted_answers, k=3)
        print(f"MAP@3 for Question ID {question_id}: {map_at_3:.4f}")
        map_at_3_sum += map_at_3

    

    # Calculate the mean MAP@3
    map_at_3 = map_at_3_sum / num_questions
    print(f"\nMean Average Precision @ 3 (MAP@3): {map_at_3:.4f}")


if __name__ == "__main__":

    # Paths to the test and prediction files
    test_file = './data/test_data_0.5k.csv'  # Replace with your test file path
    prediction_file = './output/submission.csv'  # Replace with your prediction file path

    # Evaluate the MAP@3
    evaluate_map_at_3(test_file, prediction_file)
