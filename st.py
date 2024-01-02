import re
import PyPDF2
import streamlit as st 
import requests
import nltk

nltk.download('punkt')  # You only need to run this once to download the Punkt tokenizer data

from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from Paragraph_extraction import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



def extract_text_from_pdf(pdf_path):    # this function will simply load pdf file and return the content inside the file.
        text = ""
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
            return text



def create_paragraph_from_csv(csv_path):   # We can't extract each paragraph from the documents. So this function will return a paragraph by combining the continuous rows in the csv.
        df = pd.read_csv(csv_path)
        paragraphs = []
        current_paragraph = []
        for content in df['content']:
            if pd.notna(content):  # Check if the cell is not empty
                current_paragraph.append(content)
            elif current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
        return paragraphs
#         print(paragraphs)
# create_paragraph_from_csv("V:\Citation\csv\Sony Infotainment System.csv")



# sentences = "Ldac"
# def create_document_id(sentences):
#         citation_network = link_maindoc_references(main_document_path)
#         r = [ref for cit,ref in citation_network.items()]
#         tfidf_vectorizer = TfidfVectorizer()                                                              # Vectorize the text using TF-IDF
#         tfidf_matrix = tfidf_vectorizer.fit_transform([sentences] + r)
#         cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()            # Calculate cosine similarity between the query text and each document
#         document_id = cosine_similarities.argmax()                     
#         return document_id
#         print(document_number)   
# create_sentences_with_id(main_document_path,sentences)



def create_sentences_from_csv(csv_path,doc_id):   # We can't extract each paragraph from the documents. So this function will return a paragraph by combining the continuous rows in the csv.
        paragraphs = create_paragraph_from_csv(csv_path)
        sentence_dict = {}
        document_number = doc_id
        for paragraph_number, paragraph in enumerate(paragraphs):
            sentences = paragraph.split('. ')
            for sentence_number, sentence in enumerate(sentences):
                # document_number = create_document_id(sentence)
                key = [document_number, paragraph_number, sentence_number]
                sentence_dict[tuple(key)] = sentence
        sentences=[]
        for i,j in sentence_dict.items():
            sentences.append(j+" "+ str(i))
        return sentences
        # print(sentences)
# create_sentences_from_csv("V:\Citation\csv\Sony Infotainment System.csv",10)



main_document_path = 'Sony Infotainment System.pdf'   

reference_documents_dict= {1 : "Android Auto review Everything you need to drive.pdf",
                            2 : "Anti-glare Sleek Touchscreen.pdf",
                            3 : "CarPlay The ultimate copilot.pdf",
                            4 : "Rear-View Camera.pdf",
                            5 : "What you need to know about Sony's LDAC.pdf" }

reference_documents_path= ["Android Auto review Everything you need to drive.pdf",
                           "Anti-glare Sleek Touchscreen.pdf",
                           "CarPlay The ultimate copilot.pdf",
                           "Rear-View Camera.pdf",
                           "What you need to know about Sony's LDAC.pdf" ]



citation_pattern1 = r'\[[\d\s,]+\]'
citation_pattern2 = r'\[(\d+(?:,\s*\d+)*)\]'
author_pattern1 = r'\[(?:[A-Z][a-z]*\s)+et al\.,\s\d{4}\]'
author_pattern2 = r'\[[A-Za-z\s.&]+ \d{4}\]'          #r'(?:[A-Z][a-z]*\s)+et al\.,\s\d{4}'
table_pattern = r'Table\s\d+'
figure_pattern = r'fig\s\d+'
http_pattern = r'https://\S+'

def create_paragraph_reference_dictionary(main_document_path):
        csv_path = create_csv(main_document_path)
        paragraphs = create_paragraph_from_csv(csv_path)
        pattern1 = f'({citation_pattern1}|{author_pattern1}|{figure_pattern}|{table_pattern}|{http_pattern})'
        pattern2 = f'({citation_pattern2}|{author_pattern1}|{figure_pattern}|{table_pattern}|{http_pattern})'
        paragraph_reference_dict = {}
        sentences_list = []
        for paragraph in paragraphs:
            sentences = ""
            sentences = re.split(pattern1, paragraph)
            for i in range(0, len(sentences), 2):
                if i + 1 < len(sentences) :
                    sentences_list.append(sentences[i] + sentences[i + 1])
                else:
                    sentences_list.append(sentences[i])
            citation_matches = re.findall(pattern2, paragraph)
            if citation_matches:
                for sentence in sentences_list:
                    match = re.search(pattern2, sentence)
                    if match:
                        sentence_text = sentence[:match.start()].strip()
                        citations = [c for c in match.group(1).split(',')]
                        paragraph_reference_dict[sentence_text] = ''.join(citations)
                    else:
                        paragraph_reference_dict[sentence] = "[0]"
            else:
                paragraph_reference_dict[paragraph] = "[0]"
        return paragraph_reference_dict
#         print(paragraph_reference_dict)
# create_paragraph_reference_dictionary(main_document_path)



def link_maindoc_references(main_document_path):
        pattern = r"References([\s\S]*?)(?=(?:\d+\s*\[)|(?:\Z))"
        citation_dict = {}
        current_citation = None
        references = ""
        match = re.search(pattern, extract_text_from_pdf(main_document_path))                  # Search for the "References" section and extract the content
        if match:
            references += match.group(0)
        lines = references.split('\n')
        for line in lines:
            match = re.match(r'\[(\d+)\](.*)', line)
            if match:
                current_citation = int(match.group(1))
                citation_dict[current_citation] = match.group(2).strip()
            else:
                if current_citation is not None:
                    citation_dict[current_citation] += f" {line.strip()}"
        return citation_dict
#         print(citation_dict)
# link_maindoc_references("Sony Infotainment System.pdf")



def find_similar_document(reference,reference_documents_path):
        document_texts = [extract_text_from_pdf(pdf) for pdf in reference_documents_path]       # Extract text from PDF documents
        tfidf_vectorizer = TfidfVectorizer()                                                    # Vectorize the text using TF-IDF
        tfidf_matrix = tfidf_vectorizer.fit_transform([reference] + document_texts)
        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()  # Calculate cosine similarity between the query text and each document
        most_related_document_index = cosine_similarities.argmax()                              # Find the index of the most related document
        most_related_document_path = reference_documents_path[most_related_document_index]      # Get the path of the most related document
        return most_related_document_path



def map_citation_with_reference_document(cn, reference_documents_path):  
        citation_network = link_maindoc_references(main_document_path)
        reference_dict = {}
        corresponding_document_path = find_similar_document(citation_network[cn],reference_documents_path)
        for key,value in reference_documents_dict.items():
             if value == corresponding_document_path:
                doc_id = key
        csv_path = create_csv(corresponding_document_path)
        para_list = create_sentences_from_csv(csv_path,doc_id)
        reference_dict[cn] = para_list
        return reference_dict
#         print(most_related_document_path)
# map_citation_with_reference_document(3, reference_documents_path)



def map_author_with_reference_document(reference,reference_documents_path):
        reference_dict = {}
        corresponding_document_path = find_similar_document(reference,reference_documents_path)
        for key,value in reference_documents_dict.items():
             if value == corresponding_document_path:
                doc_id = key
        csv_path = create_csv(corresponding_document_path)
        para_list = create_sentences_from_csv(csv_path,doc_id)
        reference_dict[reference] = para_list
        return reference_dict  ##{}
#         print(reference_dict)
# map_author_with_reference_document("[Rajpurkar et al., 2016]", reference_documents_path)




def map_http(http_link):
    response = requests.get(http_link)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        page_text = soup.get_text()
    reference = sent_tokenize(page_text)
    return reference
#     print(reference[0])

# map_http("https://towardsdatascience.com/attention-is-all-you-need-discovering-the-transformer-paper-73e5ff5e0634")
     
     


def similarity_search(paragraph,reference_sentences_list):
     embedding = HuggingFaceEmbeddings()
     db = FAISS.from_texts(reference_sentences_list, embedding)
     sentences = sent_tokenize(paragraph)
     lst=[]
     for sentence in sentences:
        similar_docs = db.similarity_search(sentence, k=1)
        lst.append(similar_docs[0].page_content)
     return list(set(lst))



def create_knowledge_content(main_document_path, reference_documents_path):
        paragraph_citations_dictionary = create_paragraph_reference_dictionary(main_document_path)
        for para, ref in paragraph_citations_dictionary.items():
            if re.findall(citation_pattern1,ref) and ref != "[0]":
                 cit = re.findall(citation_pattern1,ref)
                 cit = [int(c.strip("[]")) for c in cit]
                 relavant_content = map_citation_with_reference_document(cit[0], reference_documents_path)
                 result = similarity_search(para,relavant_content[cit[0]])
                 st.write(cit,para,result)
                 st.write("========================================================================================")
           
            elif re.findall(author_pattern2,ref):
                cit = re.findall(author_pattern2,ref)
                relavant_content = map_author_with_reference_document(cit[0], reference_documents_path)
                result = similarity_search(para,relavant_content[cit[0]])
                st.write(cit,para,result)
                st.write("========================================================================================")
            
            elif re.findall(http_pattern,ref):
                 cit = re.findall(http_pattern,ref)
                 relavant_content = map_http(cit[0])
                 result = similarity_search(para,relavant_content)
                 st.write(cit,para)
                 st.write("========================================================================================")

            elif re.findall(table_pattern,ref):
                 cit = re.findall(table_pattern,ref)
                 st.write(cit,para,cit)
                 st.write("========================================================================================")

            elif re.findall(figure_pattern,ref):
                 cit = re.findall(figure_pattern,ref)
                 st.write(cit,para,cit)
                 st.write("========================================================================================")

            elif re.findall(citation_pattern1,ref) and ref == "[0]":
                 st.write(para)
                 st.write("========================================================================================")

            else:
                 st.write(para,"Reference not matched ")
                 st.write("========================================================================================")
                 

create_knowledge_content(main_document_path, reference_documents_path)


