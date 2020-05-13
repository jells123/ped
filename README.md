# ped


### Notebooks gen + sync

```shell script
jupytext --to notebook notebook.py              # convert notebook.py to an .ipynb file with no outputs
jupytext --set-formats ipynb,py notebook.ipynb  # Turn notebook.ipynb into a paired ipynb/py notebook
```