# sync_data.py
import os
from huggingface_hub import HfApi

def run_sync():
    api = HfApi()
    repo = "BhaskarAnjappa/Enterprise_RAG"
    
    print("🔄 Scanning cloud workspace directory...")
    
    # 1. Fetch all files currently inside the root of your Hugging Face Space
    try:
        repo_files = api.list_repo_files(repo_id=repo, repo_type="space")
        
        # 2. Automatically identify and purge any loose PDF files uploaded to the root
        for file_path in repo_files:
            if "/" not in file_path and file_path.lower().endswith(".pdf"):
                print(f"🗑️ Found misplaced loose file. Removing: {file_path}")
                api.delete_file(
                    path_in_repo=file_path,
                    repo_id=repo,
                    repo_type="space"
                )
    except Exception as e:
        print(f"⚠️ Clean-up warning: {str(e)}")

    # 3. Stream your local data folder correctly to the /data destination path
    print("🚀 Streaming local data directory to the correct cloud path...")
    try:
        api.upload_folder(
            repo_id=repo,
            folder_path="./data",
            path_in_repo="data",
            repo_type="space"
        )
        print("✅ Success! Data directory structural matrices synchronized.")
    except Exception as e:
        print(f"❌ Upload failed: {str(e)}")

if __name__ == "__main__":
    run_sync()