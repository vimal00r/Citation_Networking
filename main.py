import streamlit as st
import pandas as pd
import csv
import comtypes
comtypes.CoInitialize()

from Paragraph_extraction_from_pdf import *
from Table_to_text.Detect_table_redact import *
from Flowchart_to_text.Detect_flowchart_redact import *
from Flowchart_to_text.flowchart_to_text import *



def create_id(doc_id,pdf_path):  
        
        table_id_dict = table_to_text(pdf_path)  # take "test_doc1.pdf" as input and give "table_to_text.pdf" as output
        flowchart_id_dict = flowchart_to_text("table_to_text_output.pdf")  # take "table_to_text_output.pdf" as input and give "final.pdf" as output
        csv_path = create_csv("flowchart_to_text_output.pdf")

        def create_paragraph_from_csv(csv_path):
            sentence_list = []
            df = pd.read_csv(csv_path)
            for content in df['text']:
                if pd.notna(content):  # Check if the cell is not empty
                    sentence_list.append(content)
            paragraphs_list = []
            index = 0 
            
            while index < len(sentence_list):
                para = ""
                if (sentence_list[index] in table_id_dict) or (sentence_list[index] in flowchart_id_dict):  # Check if the sentence is a key in the dictionary
                    para = sentence_list[index]  # Append the corresponding value directly
                    index += 1
                elif len(sentence_list[index].split()) < 15:
                    count = index
                    while True:
                        para += " " + sentence_list[count]  # Add '\n' to separate paragraphs
                        if len(sentence_list[count + 1].split()) > 15:
                            para += " " + sentence_list[count + 1]
                            count += 1
                            break
                        else:
                            count += 1
                    index = count + 1  # Update i to the next paragraph
                else:
                    para=sentence_list[index]
                    index += 1  # Update i when the paragraph has more than 15 words
                paragraphs_list.append(para)
            
            return paragraphs_list


        paragraphs = create_paragraph_from_csv("./csv/"+csv_path)
                
        sentence_dict = {}
        document_number = doc_id

        for paragraph_number, paragraph in enumerate(paragraphs, start=1):
            if table_id_dict.get(paragraph):
                sentence_dict[paragraph] = table_id_dict.get(paragraph)
                paragraph_number = paragraph_number - 1

            elif flowchart_id_dict.get(paragraph):
                sentence_dict[paragraph] = flowchart_id_dict.get(paragraph)
                paragraph_number = paragraph_number - 1 

            else:
                sentences = paragraph.split('. ')
                for chunk_number, sentence in enumerate(sentences, start=1): 
                    key = [document_number, paragraph_number, chunk_number]
                    key = f"[{document_number}.{paragraph_number}.{chunk_number}]"
                    sentence_dict[key] = sentence
        
        with open("output.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Key", "Value"])
            for key, value in sentence_dict.items():
                writer.writerow([key, value])
        
        sentences=[]
        for i,j in sentence_dict.items():
            sentences.append(j+" "+ str(i))
        return sentences

result = create_id(1,"test_doc3.pdf")
st.write(result)



# def create_paragraph_from_csv(csv_path):
#             sentence_list = []
#             df = pd.read_csv(csv_path)
#             for content in df['text']:
#                 if pd.notna(content):  # Check if the cell is not empty
#                     sentence_list.append(content)

#             paragraphs = [item for item in sentence_list if item.strip() != '']

#             paragraphs_list = [sentence for sentence in paragraphs if (len(sentence.split()) >= 15 or (sentence in table_id_dict)) or (len(sentence.split()) >= 15 or (sentence in flowchart_id_dict))]
#             return paragraphs_list