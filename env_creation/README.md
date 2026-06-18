# .venv creation
To create a virtual environment for this project simply run in the main folder:
```bash
python -m venv .venv
pip install -r requirements.txt
```
After that the environment is created paste the file `local_sources.pth` in `.venv/lib/python3.9/site-packages/`. This will allow the environment to resolve the custom modules in `sources/`.