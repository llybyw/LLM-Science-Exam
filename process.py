import argparse
import os
import pandas as pd

def process_csv(input_file, output_file):
    """
    Process a single CSV file by adding an 'id' column at the beginning.
    
    :param input_file: Path to the input CSV file.
    :param output_file: Path to the output CSV file.
    """

    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file)

    # Insert a new column 'id' at the beginning of the DataFrame
    df.insert(0, 'id', range(len(df)))

    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"New CSV file saved to: {output_file}")

def process():

    """

    Main function to handle command-line arguments and process CSV files.

    """

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Add an 'id' column to the beginning of each CSV file in the specified directory.")
    
    # Add required command-line arguments
    parser.add_argument('--data_dir', required=True, help='Directory containing the input CSV files')
    parser.add_argument('--output_dir', required=True, help='Directory where the output CSV files will be saved')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Extract the directories from the arguments
    data_dir = args.data_dir
    output_dir = args.output_dir

    # Ensure the output directory exists, create it if it doesn't
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each CSV file in the input directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            input_file = os.path.join(data_dir, filename)
            output_file = os.path.join(output_dir, filename)
            process_csv(input_file, output_file)

process()