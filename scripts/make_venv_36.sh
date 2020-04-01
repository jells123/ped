virtualenv --python=python3.6 venv36
source venv/bin/activate
pip install -r requirements_36.txt

jupyter notebook --generate-config -y
echo 'c.NotebookApp.contents_manager_class = "jupytext.TextFileContentsManager"' >> ~/.jupyter/jupyter_notebook_config.py
