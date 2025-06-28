import os
import markdown2
from bs4 import BeautifulSoup
import re
from textblob import TextBlob
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from transformers import BertTokenizer, BertModel
import numpy as np

# Loads NLTK resources (if not already downloaded)
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Loads the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Defines input directory
input_dir = "data/processed/megaset"


def preprocess_markdown(file_path):
    """
    Preprocesses a Markdown file, extracting features and BERT embeddings.

    Args:
        file_path (str): Path to the Markdown file.

    Returns:
        dict: Dictionary with extracted features.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            markdown_text = f.read()

        # Converts Markdown to HTML
        html = markdown2.markdown(markdown_text)
        cleaned_html = re.sub(r"\s+", " ", html).strip()

        # Parse content using BeautifulSoup
        soup = BeautifulSoup(cleaned_html, "html.parser")
        title = soup.find("h1").text.strip() if soup.find("h1") else ""
        abstract = (
            soup.find("p", class_="abstract").text.strip()
            if soup.find("p", class_="abstract")
            else ""
        )
        
        tokens = word_tokenize(cleaned_html)
        sentence_count = len(sent_tokenize(cleaned_html)),
        avg_sentence_length = len(tokens) / max(1, len(sent_tokenize(cleaned_html))),

        return (
            cleaned_html,
            len(title),
            len(abstract),
            sentence_count,
            avg_sentence_length,
            int(bool(abstract))
        )

    except Exception as e:
        raise RuntimeError(f"Error processing {file_path}: {e}")


# Function to extract features from Markdown content
def extract_features(
    markdown_text,
    title_length,
    abstract_length,
    sentence_count,
    avg_sentence_length,
    has_abstract,
):
    """
    Extracts features from Markdown text.

    Args:
        markdown_text (str): The Markdown text.

    Returns:
        list: A list of extracted features.
    """

    # 1. Basic Text Features
    words = re.findall(r"\b\w+\b", markdown_text.lower())  # Extract words
    word_count = len(words)
    unique_words = len(set(words))

    # 2. Section Counts (example)
    section_counts = {}
    for section in ["Introduction", "Methods", "Results", "Discussion", "Conclusion"]:
        section_counts[section] = markdown_text.count(section)

    # 3. Keyword Counts (example)
    keywords = ["significant", "novel", "innovative", "impact", "findings"]
    keyword_counts = sum([markdown_text.lower().count(keyword) for keyword in keywords])

    # 4. Sentiment Analysis
    sentiment = TextBlob(markdown_text).sentiment.polarity

    # 5. Text Length
    text_length = len(markdown_text)

    # 6. Vocabulary Richness (Type-Token Ratio)
    type_token_ratio = len(set(words)) / len(words)

    # 7. Stop Word Count
    stop_words = set(stopwords.words("english"))
    stop_word_count = len([w for w in words if w.lower() in stop_words])

    # 8. Punctuation Count
    punctuation_count = len([c for c in markdown_text if c in string.punctuation])

    # 9. Stemmed Word Count (basic stemming)
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]
    stemmed_word_count = len(stemmed_words)

    # 10. BERT Embeddings
    inputs = tokenizer(
        markdown_text, return_tensors="pt", padding=True, truncation=True
    )
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

    # Combine features into a list
    features = [
            np.array(word_count),
            np.array(sentence_count),
            np.array(avg_sentence_length),
            np.array(unique_words),
            np.array(type_token_ratio),
            np.array(stop_word_count),
            np.array(punctuation_count),
            np.array(stemmed_word_count),
            np.array(sentiment),
            np.array(title_length),
            np.array(has_abstract),
            np.array(abstract_length),
            np.array(text_length),
            np.array(list(section_counts.values())),
            np.array([keyword_counts]),
            np.array(embeddings.flatten())
    ]

    return features


def process_directory(data_dir, output_dir):
    """
    Processes Markdown files in a directory and saves extracted features as .npy files.

    Args:
        data_dir (str): Path to the directory containing Markdown files.
        output_dir (str): Path to the directory to save processed features.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(data_dir):
        if filename.endswith(".md"):
            try:
                file_path = os.path.join(data_dir, filename)
                a,b,c,d,e,f = preprocess_markdown(file_path)
                features = extract_features(a,b,c,d,e,f)

                # Flattens the features and ensures homogeneous shape
                feature_values = []
                for value in features:
                    if isinstance(value, np.ndarray):
                        feature_values.extend(value.flatten())
                    else:
                        feature_values.append(value)

                output_file = os.path.join(output_dir, filename.replace(".md", ".npy"))

                # Saves the feature values to the .npy file
                np.save(output_file, feature_values)
            except Exception as e:
                print(f"Error processing {filename}: {e}")


# Main Function
if __name__ == "__main__":
    data_dir = "data/interim/dataset/Markdown"  # Adjust the path if necessary
    output_dir = "data/processed/dataset"
    process_directory(data_dir, output_dir)
