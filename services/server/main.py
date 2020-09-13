
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 15:53:10 2020
@author: deviantpadam
"""

from flask import Flask, jsonify, request, render_template
from torch2vec.torch2vec import LoadModel

app = Flask(__name__)
model = LoadModel('../train/weights.npy',f_size=3,pad=[4,2,1])

@app.route('/')
def home():
    doc_ids = model.docids
    if not request.args.get('num'):
        num = 0
    else: 
        num = int(request.args.get('num'))
    return render_template('home.html', doc_ids = doc_ids[25*num:25*(num+1)], num=num)

@app.route('/torch')
def torch():
    id = request.args.get('id')
    sim = model.similar_docs(int(id),topk=10)
    similarity = []
    for i in range(len(sim[0])):
        #if sim[1][i] == NaN:
        #   sim[1][i] = 0.0
        similarity.append(dict(id=sim[0][i])) #, score=sim[1][i])) #working
    return jsonify(similarity)


if "__main__"==__name__:
    app.run(host="0.0.0.0", port="5002")
