
![Alt Text](drug.jpg)

# Drug-review-sentiment-analysis
This project analyzes patient drug reviews to classify their sentiment as Positive or Negative using a fine-tuned DistilBERT model from Hugging Face. Built with Streamlit for an interactive and user-friendly web interface.

## 🔍 Features

- Input a single drug review and get instant sentiment prediction
- Confidence score of the model prediction
- Clean, interactive Streamlit web app interface
- Uses Hugging Face Transformers (`distilbert-base-uncased-finetuned-sst-2-english`)

## 🚀 Tech Stack

- Python 🐍
- Hugging Face Transformers 🤗
- DistilBERT (fine-tuned on SST-2) 🧠
- Streamlit 🌐
- Pandas (for potential data handling) 📊

## 📦 How to Run

```bash
git clone https://github.com/your-username/drug-review-sentiment-analysis.git
cd drug-review-sentiment-analysis
pip install -r requirements.txt
streamlit run app.py
