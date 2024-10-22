# TextRanking-Algo

# Text Summarization Using Flask and NLTK

## Overview

This project is a web application for text summarization using Python, Flask, BeautifulSoup, and NLTK. The application allows users to input a wikipedia URL from which the text will be extracted and summarized. The project demonstrates the use of Natural Language Processing (NLP) techniques for text summarization and is built to be a simple and effective tool for summarizing large articles.


## Steps

1. **Clone the repository**:

    ```bash
    git clone https://github.com/varuniherle/TextRanking-Algo.git
    ```

2. **Create a virtual environment**:

    ```bash
    python -m venv venv
    ```

3. **Activate the virtual environment**:

    - On Windows:

        ```bash
        venv\Scripts\activate
        ```

    - On macOS/Linux:

        ```bash
        source venv/bin/activate
        ```

4. **Install the dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

5. **Run the application**:

    ```bash
    flask run
    ```

    The application will be available at `http://127.0.0.1:5000`.

## Usage

1. Open your web browser and navigate to `http://127.0.0.1:5000`.
2. Enter the URL of the article you want to summarize.
3. Click the "Summarize" button to get the summarized text.

