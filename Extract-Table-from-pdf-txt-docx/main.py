# %%
import pandas as pd
import os
import tabula
from docx.api import Document

# %%

if os.path.isdir('Parent')== True:
    os.chdir('Parent')
#FOR CHILD1 DIRECTORY
if os.path.isdir('Child1')==True:
    os.chdir('Child1')
#PDF FILE READING
if os.path.isfile('Pdf1_Child1.pdf')==True:
    df_pdf_child1=tabula.read_pdf('Pdf1_Child1.pdf',pages='all')
#DOCUMENT READING
if os.path.isfile('Document_Child1.docx')==True:
    document = Document('Document_Child1.docx')
    table = document.tables[0]
    data = []

    keys = None
    for i, row in enumerate(table.rows):
        text = (cell.text for cell in row.cells)
        if i == 0:
            keys = tuple(text)
            continue
        row_data = dict(zip(keys, text))
        data.append(row_data)
df_document_child1=pd.DataFrame(data)
#TEXT READING
if os.path.isfile('Text_Child1.txt')==True:
    df_text_child1=pd.read_csv('Text_Child1.txt')

# %%
df_text_child1


# %%
os.chdir('../')
if os.path.isdir('Parent')== True:
    os.chdir('Parent')
#FOR CHILD2 DIRECTORY
if os.path.isdir('Child2')==True:
    os.chdir('Child2')
#PDF FILE READING
if os.path.isfile('Pdf1_Child2.pdf')==True:
    df_pdf_child2=tabula.read_pdf('Pdf1_Child2.pdf',pages='all')
#DOCUMENT READING
if os.path.isfile('Document_Child2.docx')==True:
    document = Document('Document_Child2.docx')
    table = document.tables[0]
    data = []

    keys = None
    for i, row in enumerate(table.rows):
        text = (cell.text for cell in row.cells)
        if i == 0:
            keys = tuple(text)
            continue
        row_data = dict(zip(keys, text))
        data.append(row_data)
df_document_child2=pd.DataFrame(data)
#TEXT READING
if os.path.isfile('Text_Child2.txt')==True:
    df_text_child2=pd.read_csv('Text_Child2.txt')

# %%
df_pdf_child2[0].head(4)

# %%
os.chdir('../')
if os.path.isdir('Parent')== True:
    os.chdir('Parent')
#FOR CHILD3 DIRECTORY
if os.path.isdir('Child3')==True:
    os.chdir('Child3')
#PDF FILE READING
if os.path.isfile('Pdf1_Child3.pdf')==True:
    df_pdf_child3=tabula.read_pdf('Pdf1_Child3.pdf',pages='all')
#DOCUMENT READING
if os.path.isfile('Document_Child3.docx')==True:
    document = Document('Document_Child3.docx')
    table = document.tables[0]
    data = []

    keys = None
    for i, row in enumerate(table.rows):
        text = (cell.text for cell in row.cells)
        if i == 0:
            keys = tuple(text)
            continue
        row_data = dict(zip(keys, text))
        data.append(row_data)
df_document_child3=pd.DataFrame(data)
#TEXT READING
if os.path.isfile('Text_Child3.txt')==True:
    df_text_child3=pd.read_csv('Text_Child3.txt')

# %%
df_text_child3

# %%
