from flask import Flask, request, render_template
import tokenizer
from stemmer import Stemmer
from utils.sentiment import positive_data,negative_data,guj_stopwords
from utils.stopwords import stopwords

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form_post():

    given_text=request.form['guj_text']
    
    # Sentence Tokenization
    sent_tokens=tokenizer.SentenceTokenizer(given_text)
    print("After Sentence Tokenization")
    print(sent_tokens)

    # Word Tokenization
    word_tokens=tokenizer.WordTokenizer(given_text,keep_punctuations=False) 
    print("After Word Tokenization")
    print(word_tokens)

    #Stemming
    stemmer = Stemmer()
    stemmed_text=stemmer.stem(given_text)
    print("After Stemming: ")
    print(stemmed_text)

    #Stopword Removal
    stopword_removal=[]
    if len(sent_tokens) > 1:
        for sent in stemmed_text:
            stopword_removal.append(tokenizer.WordTokenizer(sent,keep_punctuations=False,keep_stopwords=False)) 
    else:
        stopword_removal.append(tokenizer.WordTokenizer(stemmed_text,keep_punctuations=False,keep_stopwords=False))
    print("After Stop Word Removal")
    print(stopword_removal)

    pos_cnt=0
    neg_cnt=0
    neutral_cnt=0
    tot_words=0

    input_sentence=stemmed_text
    for sent in input_sentence:
        list_items=input_sentence
        if " " in sent:
            list_items=sent.split(" ")
        for word in list_items:
            if word not in stopwords:
                tot_words+=1
                if word in positive_data:
                    pos_cnt+=1
                    continue
                if word in negative_data:
                    neg_cnt+=1
                    continue
                neutral_cnt+=1

    
    
    if (pos_cnt>neg_cnt or pos_cnt>neutral_cnt):
        final_verdict="Positive"
    elif (neg_cnt>pos_cnt or neg_cnt>neutral_cnt):
        final_verdict="Negative"
    else:
        final_verdict="Neutral"

    return render_template('form.html', given_text=given_text,final_verdict=final_verdict,pos=round(pos_cnt/tot_words,2),
    neg=round(neg_cnt/tot_words,2),neutral=round(neutral_cnt/tot_words,2),
    sent_tokens=sent_tokens,word_tokens=word_tokens,stemmed_text=stemmed_text,stopword_removal=stopword_removal)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)
