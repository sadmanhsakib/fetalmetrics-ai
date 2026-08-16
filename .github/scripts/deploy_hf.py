from huggingface_hub import HfApi
import os

from dotenv import load_dotenv

load_dotenv()

hf_readme = """---
title: FetalMetrics AI
emoji: 👶
colorFrom: green
colorTo: blue
sdk: streamlit
app_file: src/app.py
pinned: true
---
"""

requirements = """folium
geopandas
joblib
matplotlib
pandas
shap
streamlit
streamlit-folium"""

with open("HF_README.md", "w") as f:
    f.write(hf_readme)

with open("requirements.txt", "w") as f:
    f.write(requirements)

token = os.getenv("HF_TOKEN")
api = HfApi()
repo_id = "sadmanhsakib/fetalmetrics-ai"

# Upload only required folders
for folder, repo_path in [
    ("src", "src"),
    ("models", "models"),
    (".streamlit", ".streamlit"),
]:
    api.upload_folder(
        folder_path=folder,
        path_in_repo=repo_path,
        repo_id=repo_id,
        repo_type="space",
        token=token,
    )

# Upload README and requirements
for local, remote in [
    ("HF_README.md", "README.md"),
    ("requirements.txt", "requirements.txt"),
]:
    api.upload_file(
        path_or_fileobj=local,
        path_in_repo=remote,
        repo_id=repo_id,
        repo_type="space",
        token=token,
    )
