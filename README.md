## Setup ##
In PyCharm, navigate to `File(top left) > Settings > Project: <Project name> > Python Interpreter > Create a new Interpreter (with virtual env)`.

When done, a directory called `venv` will appear (which is ignored by git, see .gitignore file). 

Then use the following command to fetch all the required dependencies into the virtual env.

Mac/Linux:
```shell
venv/bin/python -m pip install -r requirements.txt
```
Windows:
```shell
venv\bin\python -m pip install -r requirements.txt
```

### Run ###
After installing requirements, you can use the run icon in PyCharm to run the project.

Bush bush bushi
