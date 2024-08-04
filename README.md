import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

def main():
    sid = SentimentIntensityAnalyzer()
    text = input("Enter a sentence: ")

    scores = sid.polarity_scores(text)

    if scores['compound'] >= 0.05:
        sentiment = "positive"
    elif scores['compound'] <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    print(f"Sentiment: {sentiment}")

if __name__ == "__main__":
    main()
- 👋 Hi, I’m @DOUGLASDAVIS
- 👀 I’m interested in ... yarn add chatgpt && yarn install
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...

<!---
ELONISEVIL/ELONISEVIL is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
