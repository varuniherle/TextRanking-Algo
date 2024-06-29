import bs4 as bs
import urllib.request
import re
import nltk
import heapq

# Download required NLTK packages
nltk.download('punkt')
nltk.download('stopwords')

def text_summarizer(url):
    # Reading the content
    try:
        scraped_data = urllib.request.urlopen(url)
        article = scraped_data.read()
    except Exception as e:
        print(f"Error fetching the URL: {e}")
        return ""  # Return empty string in case of an error

    # Parsing the article
    parsed_article = bs.BeautifulSoup(article, 'lxml')
    paragraphs = parsed_article.find_all('p')
    article_text = ""
    for p in paragraphs:
        article_text += p.text

    # Removing Square Brackets and Extra Spaces
    # \[.*?\] matches anything within square brackets (non-greedy)
    article_text = re.sub(r'\[.*?\]', ' ', article_text)
    article_text = re.sub(r'\s+', ' ', article_text)  # Replaces multiple spaces with a single space

    # Removing special characters and digits
    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text)
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

    # Tokenize into sentences
    sentence_list = nltk.sent_tokenize(article_text)

    # Calculate word frequencies
    stopwords = nltk.corpus.stopwords.words('english')
    word_frequencies = {}
    for word in nltk.word_tokenize(formatted_article_text):
        word = word.lower()  # Convert word to lowercase
        if word not in stopwords:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / maximum_frequency

    # Calculate sentence scores
    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies:
                if len(sent.split(' ')) < 30:  # Limit sentence length to less than 30 words
                    if sent not in sentence_scores:
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    # Generating summary
    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    return summary

# Test the summarizer
url = 'https://en.wikipedia.org/wiki/Reinforcement_learning'
print(text_summarizer(url))
