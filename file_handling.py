import pandas as pd
from anthropic import Anthropic
from utils import sanitize_filename

def pdf_read(client:Anthropic,path,name_doc):
    
    clean_filename = sanitize_filename(name_doc)
    
    with open(path, 'rb') as f:
        response = client.beta.files.upload(
            file=(clean_filename, f, 'application/pdf')
        )
    return response
    

