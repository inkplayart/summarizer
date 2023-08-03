"""
Summarize an entire folder's worth of documents. Requires langchain, transformers, torch, and I forget if it's pypdf or pypdf2 or some such. Basically, try to run it and install whatever it tells you to.


Set up your folder structure like this:

- Put this file into a main folder, say "summarizer"
- Create a folder, "papers", where all the raw pdfs go
- Creat a folder, "results", where all the processed text files go

"""

import os
import sys
from transformers import AutoTokenizer
import transformers
import torch
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import traceback

#Whether this is absolutely necessary is a matter for debate, but I'm following the hf tutorial
system_prompt_page = """<s>[INST] <<SYS>>
You are working on detailed and comprehensive overviews of pages of a document. Review the following text, and provide an overview of what you've read that is at least 400 words long. The text is: 
<</SYS>>"""

system_prompt_running = """<s>[INST] <<SYS>>
You are working on creating a running summary of a document. You will be provided with two pieces of text. The first piece of text is the running summary of what you've read so far. The second is a summary of the current page. Please update the running summary by adding any new information from the page to the running summary. The running summary is: " 
<</SYS>>"""

system_prompt_cleanup = """<s>[INST] <<SYS>>
 clean up the following text. It contains information about a longer document, but it is disjointed. I want a single, coherent text that includes all of the details and is at 1000 words long, but doesn't talk about a specific page or repeat itself." 
<</SYS>>"""

def extract_tokens(text,tokenizer):
    numToks = len(tokenizer(text).input_ids)
    #print(numToks)
    return numToks

def concatenate_strings(docs, max_length,tokenizer):
    result = []
    current_string = ""
    strings = []
    for doc in docs:
        strings.append(doc.page_content)

    for s in strings:
        if extract_tokens(current_string,tokenizer) + extract_tokens(s,tokenizer) <= max_length:
            current_string += s
        else:
            result.append(current_string)
            current_string = s

    result.append(current_string)
    return result


"""
Processes pdf or docx files only. Not .txt. Extract a single page, then analyze it, and output your analysis, repeat for all pages
"""
def extract_and_analyze_from_docs(input_filename,pipeline,tokenizer,output_filename=None,sourceDir='.',destinationDir='.',query=""):
    #load the document
    print("Extract: ", os.path.join(sourceDir,input_filename))
    loader = PyPDFLoader(os.path.join(sourceDir,input_filename))
    txt_docs = loader.load_and_split()
    
    #load_and_split() #split on page
    if txt_docs != []:
        newDocs = concatenate_strings(txt_docs,3000,tokenizer)
        analysis = ""
        i = 1
        for page_num, txt in enumerate(newDocs):
            #Step 1: Current page
            prompt = system_prompt_page + txt + " \n Write your summary now. [/INST]"
            sequences = pipeline(
            prompt,
            do_sample=True,
            top_k=3,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=768
            )
            for seq in sequences:
                model_out = seq['generated_text'].split("[/INST]")[-1].strip()
            
            #Step 2: Running summary
            prompt = system_prompt_running + analysis + " \n The detailed summary of the current page is: \n" + model_out + "\n Please update the running summary by adding any new information from the page to the running summary. Remember, it should contain around 1000 words and give the reader a really good understanding of what is in the whole document. [/INST]"
            sequences = pipeline(
            prompt,
            do_sample=True,
            top_k=3,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=3000
            )
            for seq in sequences:
                model_out = seq['generated_text'].split("[/INST]")[-1].strip()
                
            #Step 3: Cleanup
            prompt = system_prompt_cleanup + model_out + " \n Write the cleaned up version now. [/INST]"
            sequences = pipeline(
            prompt,
            do_sample=True,
            top_k=3,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=3000
            )
            for seq in sequences:
                model_out = seq['generated_text'].split("[/INST]")[-1].strip()
            analysis = model_out
            with open(str(i)+".txt",'w',encoding='utf-8',errors='ignore') as wr:
                wr.write(analysis)
            i = i+1
            print("Processed page......", str(page_num))
        txtname = input_filename.split(".")[0] + "_processed.txt"
        if output_filename is not None:   
            txtname = output_filename
        with open(os.path.join(destinationDir,txtname),'w',encoding='utf-8',errors='ignore') as writer:
            writer.write(analysis)
            if analysis == "":
                writer.write("NOT FOUND")
        print("Done extraction")
    else:
        print("Error with ", input_filename, '\n')


#I've tested 7B and 13B. There isn't a HUGE difference. I've found 7B is quite good and faster, but you do you. Use 70B if you've got the VRAM...
model =  "meta-llama/Llama-2-7b-chat-hf"

#load the model
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

for file in os.listdir("papers"):
    try:
        extract_and_analyze_from_docs(file,pipeline,tokenizer,output_filename = None,sourceDir="papers",destinationDir="results")
    except:
        print("Error with " + file)
    