import PyPDF2
import os
from constants import *
import string
import json
import re
from difflib import SequenceMatcher
from spacy_langdetect import LanguageDetector
from spacy.language import Language
import spacy
import pandas as pd
from multiprocessing import Pool




def isValidOrder(text: str):
    abstractFound = False
    for i in range(len(text)):
        if not abstractFound and 'abstract' in text[0:i]:
            abstractFound = True
        if 'introduction' in text[0:i]:
            if abstractFound:
                return True
            return False
        # references showing up before "abstract" and "introduction"
        if 'references' in text[0:i]:
            return False
        

def isEnglish(sentence: str):
    nlp = spacy.load('en')
    nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
    doc = nlp(sentence)
    detect_language = doc._.language
    if detect_language["language"] != 'en':
        return False
    return True

rootdir = '../pollack/articles/arxiv' # replace with your path

validArticles = {}
for subdir, dirs, files in os.walk(rootdir):
    subdir = subdir.replace('\\', '')
    subdir = subdir.replace(f'{rootdir}', '')
    # print('reading folder: ' + subdir)
    counter = 0
    for file in files:
        # im only taking the first 10 valid articles for each folder here so I can show input and output on github
        if counter == 10:
            break
        with open(f'{rootdir}/{subdir}/{file}', 'r', encoding='cp1252', errors='ignore') as f:
            text = " ".join(f.readlines())
            # only look for articles with "abstract", "introduction", and "references" to make cleaning possible
            if 'abstract' in text.lower() and 'introduction' in text.lower() and 'references' in text.lower():
                # need to make sure the show up in the right order
                if isValidOrder(text.lower()):
                    validArticles[file] = text 
                    counter += 1


def cleanArticle(text: str):
    cleanedText = ""
    startingIndex = text.lower().find('introduction') + len('introduction')
    endingIndex = text.lower().rfind('references')

    # only read from the article's introduction section to its references section 
    text = text[startingIndex:endingIndex]

    # get rid of numbers, keep punctuation
    text = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)

    #     The \( and \) say we want to target the actual parenthesis.
    #     Then the parenthesis around the expression (.*?) say that we want to group what is inside
    #     Finally the .*? means that we want any character . and any repetition of that character *?.
    text = re.sub(r"\((.*?)\)", "", text)

    # remove extra spaces
    cleanedText = re.sub(' +', ' ', text)
    return cleanedText


def process(article_path: str):
    clean_path = f'../pollack/articles/cleaned/{article_path}'
    with open(clean_path, 'w') as f:
        text = cleanArticle(validArticles[article_path])
        f.write(text)
    

def main():
    
    if not os.path.exists('../pollack/articles/cleaned'):
        os.makedirs('../pollack/articles/cleaned')


    with Pool(processes=4) as p:
        p.map(process, list(validArticles.keys()))

if __name__ == '__main__':
    main()