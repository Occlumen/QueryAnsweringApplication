Documentation on setup of the application

1) Download or clone the entire application provided on the git repository.
2) Create a folder model inside Assignment and mixtral-8x7b-v0.1.Q4_K_M.gguf and llama-2-7b.Q4_K_S.gguf
3) create a virtual environment using python -m venv .\path\venv
4) Install all the packages required by typing
                 pip install -r requirements.txt
5) If no errors detected, then run the below given command in terminal
                   uvicorn app:app --reload
6) Go to your deafult browser and search
                 http://localhost:8000 or the link visible after step 5
7) Enter your query in the input box provided and click on submit and wait for the results.

If you want to add more documents for retrieval or change the existing one, then add the pdf to the data folder in assignment and edit the code in ingest.py. path_doc is a dictionary , so you can add more documents by just seperating them using comma.

The frontend code can be changed by editing the index.html file in frontend folder.