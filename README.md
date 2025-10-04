# LocalAIAgentWithRAG
How to run
Download the files
In CMD
Create vitual environment >python -m venv venv
Activate the environment >venv\Scripts\Activate
cd to the path where the folder is saved C:\Users\name\Downloads\WIL_RAG
Install the requirments.txt file >pip install -r requirements.txt

Download ollama and install
https://ollama.com/download

In cmd type ollama to check whether it is properly installed
ollama

If properly installed you should see the following

-----------------------------------------------------------------------------
Usage:
  ollama [flags]
  ollama [command]

Available Commands:
  serve       Start ollama
  create      Create a model
  show        Show information for a model
  run         Run a model
  stop        Stop a running model
  pull        Pull a model from a registry
  push        Push a model to a registry
  signin      Sign in to ollama.com
  signout     Sign out from ollama.com
  list        List models
  ps          List running models
  cp          Copy a model
  rm          Remove a model
  help        Help about any command

Flags:
  -h, --help      help for ollama
  -v, --version   Show version information

Use "ollama [command] --help" for more information about a command.

-----------------------------------------------------------------------------
Pulling models
Type in cmd
*Model 01
ollama pull llama3.2

*Model 02
Pulling embedding model

ollama pull mxbai-embed-large

main.py
Running the ollama model
Getting the promt

vector.py
Retriving the data

intalling streamlit
pip install streamlit
pip install matplotlib

To run the code and open the chat
streamlit run app.py

