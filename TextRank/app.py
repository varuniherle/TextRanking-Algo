from flask import Flask, request, render_template
import bs4 as bs
import urllib.request
import re
import nltk
import heapq

# Download required NLTK packages
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

def text_summarizer(url):
    try:
        scraped_data = urllib.request.urlopen(url)
        article = scraped_data.read()
    except Exception as e:
        return f"Error fetching the URL: {e}"

    parsed_article = bs.BeautifulSoup(article, 'lxml')
    paragraphs = parsed_article.find_all('p')
    article_text = ""
    for p in paragraphs:
        article_text += p.text

    article_text = re.sub(r'\[.*?\]', ' ', article_text)
    article_text = re.sub(r'\s+', ' ', article_text)

    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text)
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

    sentence_list = nltk.sent_tokenize(article_text)

    stopwords = nltk.corpus.stopwords.words('english')
    word_frequencies = {}
    for word in nltk.word_tokenize(formatted_article_text):
        word = word.lower()
        if word not in stopwords:
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / maximum_frequency

    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies:
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores:
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    return summary

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form['url']
        summary = text_summarizer(url)
        return render_template('index.html', url=url, summary=summary)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
