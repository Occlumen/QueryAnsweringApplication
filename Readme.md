Documentation on setup of the application

1) Download or clone the entire application provided on the git repository.
2) Create a folder model inside Assignment and place the downloaded model mixtral-8x7b-v0.1.Q4_K_M.gguf inside the model folder.
3) create a virtual environment using python -m venv ./venv.
4) Activate the virtual environment by using following commands in terminal of v s code
                     cd venv
                     cd Scripts
                     ./activate
5) Return to the earlier path by using following commands in terminal of v s code
                     cd ..
                     cd ..
6) create a file .env inside assignment and create a variable OPENAI_API_KEY and assign the Openi API Key.
   Make sure your openai account does have the usage limit.
7) Install all the packages required inside your virtual environment by typing 
                 pip install -r requirements.txt
8) If no errors detected while installing requirements, then run the below given command in terminal , to start running the application
                   uvicorn app:app --reload
9) Once you see Application startup complete in terminal, Go to your deafult browser and search
                 http://localhost:8000 or the link visible after step 5
10) Enter your query in the input box provided and click on submit and wait for the results.
   The results take time depending up the cpu and gpu availability(GPU has not been enabled in the code).
