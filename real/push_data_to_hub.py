import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_tag

# python push_to_hub.py \
#   --data_dir /Users/jack/Desktop/dummy_ctrl/datasets/pick_cube_20demos \
#   --repo_id JackYuuuu/test \
#   --tag v1.0

def push_dataset_to_hub(
    data_dir: Path,
    repo_id: str,
    private: bool = False,
    tag: str = None
):
    """Push dataset to HuggingFace Hub"""
    api = HfApi()
    
    # Create repository
    api.create_repo(
        repo_id=repo_id,
        private=private,
        repo_type="dataset",
        exist_ok=True
    )
    
    # Upload data directory
    api.upload_folder(
        folder_path=data_dir / "data",
        path_in_repo="data",
        repo_id=repo_id,
        repo_type="dataset"
    )
    
    # Upload videos directory
    api.upload_folder(
        folder_path=data_dir / "videos",
        path_in_repo="videos",
        repo_id=repo_id,
        repo_type="dataset"
    )
    
    # Upload meta directory
    api.upload_folder(
        folder_path=data_dir / "meta",
        path_in_repo="meta",
        repo_id=repo_id,
        repo_type="dataset"
    )
    
    print(f"Successfully pushed dataset to {repo_id}")
    
    # Create tag if specified
    if tag:
        create_tag(
            repo_id=repo_id,
            tag=tag,
            repo_type="dataset"
        )
        print(f"Created tag '{tag}' for dataset {repo_id}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Directory containing your dataset"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g. 'username/dataset-name')"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private"
    )
    parser.add_argument(
        "--tag",
        type=str,
        help="Create a tag for this version (e.g., 'v1.0', '1.0')"
    )
    
    args = parser.parse_args()
    push_dataset_to_hub(**vars(args))

if __name__ == "__main__":
    main()