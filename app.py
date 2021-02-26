# Imports
import pandas as pd
import numpy as np
import tensorflow_hub as hub

# Streamlit
import streamlit as st
import preshed
import cymem

# PDF
import sys
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import XMLConverter, HTMLConverter, TextConverter
from pdfminer.layout import LAParams
import io

# Summarization using extractive bert
from summarizer import Summarizer

st.set_page_config(layout="wide")
file, text, q = None, None, None


@st.cache()
def load_bert():
    return hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

@st.cache(hash_funcs={preshed.maps.PreshMap:id, cymem.cymem.Pool:id}, allow_output_mutation=True)#hash_funcs={preshed.maps.PreshMap: lambda x: 1, cymem.cymem.Pool:      lambda x: 1})
def load_summarizer():
    return Summarizer()

@st.cache()
def load_pdf(file)->str:
    
    if isinstance(file, str):
        fp = open(file, 'rb')
    else: 
        fp = file
        
    rsrcmgr = PDFResourceManager()
    retstr = io.StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    
    # Create a PDF interpreter object.
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    
    # Process each page contained in the document.
    pages = []
    for i, page in enumerate(PDFPage.get_pages(fp)):
        interpreter.process_page(page)
        text = retstr.getvalue()
        pages.append(text)
        
    full_text = pages[-1]
    return full_text

@st.cache()
def get_articles(text:str)->pd.Series:
    
    data = pd.Series(text.split('\n\nARTICLE'))

    s = (data
         .str.strip()
         .loc[46:]
         .loc[lambda x: x.astype(bool)]
         .loc[lambda x: x.apply(len)>10]
         .str.replace('\s+',' ')
         .drop_duplicates()
        )
    
    return s

@st.cache()
def get_embedding(s:pd.Series, model)->pd.DataFrame:
    X = s.apply(lambda x: model([x])[0]).apply(pd.Series)
    return X

### APP
st.title('BERT Passage Scoring')

# ALWAYS
model = load_bert()

if st.checkbox('Load Summarizer'):
    summarizer_model = load_summarizer()

file = st.file_uploader('Upload your document.')

if st.checkbox('Use Brexit trade deal'):
    file = 'DRAFT_UK-EU_Comprehensive_Free_Trade_Agreement.pdf'

# RUN
if file and not text:
    text = load_pdf(file)

if text:
    s = get_articles(text)
    X = get_embedding(s, model)

@st.cache()
def ask(q:str, X:pd.DataFrame, s:pd.Series, n: int, model)->pd.Series:
    
    embedding = np.array(model([q])[0])
    sorted_index = (X
                    .apply(lambda row: np.dot(row, embedding), axis=1)
                    .abs()
                    .sort_values(ascending=False)
                   )
    
    return s.loc[sorted_index.index].head(n)

#@st.cache()
def summarize(text, model, n=1):
    result = model(text, num_sentences=n)
    return result


if text:
    q = st.text_input('What is your query?')
    ans = ask(q, X=X, s=s, n=3, model=model)

if q:
    for i, t in enumerate(ans):
        with st.beta_expander(f'ARTICLE {t.split()[0]}'):
            if len(t.split('.'))>3:
                summary = summarize(t, summarizer_model, 1)
                st.success(summary)

            st.write(t)


