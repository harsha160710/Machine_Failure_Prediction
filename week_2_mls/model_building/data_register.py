from huggingface_hub import HfApi,create_repo
from huggingface_hub.utils import RepositoryNotFoundError,HfHubHTTPError
import os

repo_id="Harsha1001/Machine-Failure-Prediction"
repo_type='dataset'

api=HfApi(token=os.getenv("HF_TOKEN"))

try:
  api.repo_info(repo_id=repo_id,repo_type=repo_type)
  print(f"Spcae '{repo_id}' doesn't exists.Using it")
except RepositoryNotFoundError:
  print("Creating new space")
  api.create_repo(repo_id=repo_id,repo_type=repo_type,private=False)
  print(f"Space '{repo_id}' is created")

api.upload_folder(
    folder_path='week_2_mls/data',
    repo_id=repo_id,
    repo_type=repo_type,
)
