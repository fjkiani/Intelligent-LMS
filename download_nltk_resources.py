import nltk

print("Downloading NLTK resources...")
nltk.download('punkt')
nltk.download('punkt_tab', raise_on_error=False)  # Try to download punkt_tab specifically
nltk.download('stopwords')
print("Download complete!") 