import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm  # Import tqdm

# Check if CUDA (GPU) is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the summarization model and tokenizer
model_name = 'sshleifer/distilbart-cnn-12-6'  # Smaller model for summarization
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Move the model to the device (GPU or CPU)
model.to(device)

# Function to generate summary
def generate_summary(text):
    # Check if the text is valid
    if pd.isna(text) or not isinstance(text, str) or text.strip() == '':
        return 'No content available to summarize.'
    
    # Encode the text and move to device
    inputs = tokenizer.encode(
        "summarize: " + text,
        return_tensors='pt',
        max_length=1024,  # Model's max input length
        truncation=True
    ).to(device)
    
    # Generate summary
    summary_ids = model.generate(
        inputs,
        max_length=200,  # Adjust for desired summary length
        min_length=50,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    
    # Decode and return the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Load the CSV file into a DataFrame
df = pd.read_csv('indian_news_final.csv')

# Display the DataFrame columns for verification
print("DataFrame columns:", df.columns)

# Initialize tqdm with Pandas
tqdm.pandas()

# Apply the summary function to the 'Content' column with progress bar
print("Generating summaries...")
df['Summary'] = df['Content'].progress_apply(generate_summary)

# Save the updated DataFrame to a new CSV file
df.to_csv('indian_news_with_summaries_final.csv', index=False)

print("Summaries have been generated and saved to 'indian_news_with_summaries_final.csv'.")
