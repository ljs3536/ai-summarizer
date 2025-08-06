from flask import Flask, request, render_template, jsonify
from transformers import pipeline, PreTrainedTokenizerFast, BartForConditionalGeneration
import torch

app = Flask(__name__)

summarizer = pipeline("summarization")

# KoBart 모델 및 토크나이저 로딩
tokenizer = PreTrainedTokenizerFast.from_pretrained("digit82/kobart-summarization")
model = BartForConditionalGeneration.from_pretrained("digit82/kobart-summarization")

@app.route("/", methods = ['GET', 'POST'])
def index():
    summary = ''
    if request.method == 'POST':
        input_text = request.form['input_text']
        result = summarizer(input_text, max_length = 130, min_length = 30, do_sample = False)
        summary = result[0]['summary_text']
    return render_template('index.html', summary=summary)

@app.route("/summary", methods = ['GET', 'POST'])
def summarize():
    text = request.form['input_text']
    summary = ''
    if request.method == 'POST':
        # 입력 토크나이징
        inputs = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
        # 요약 생성
        summary_ids = model.generate(inputs, max_length=128, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return render_template('index.html', summary=summary)

if __name__=='__main__':
    app.run(debug=True)