import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi, create_tag
from datetime import datetime
from jinja2 import Template

# python push_to_hub.py \
#   --data_dir /Users/jack/Desktop/dummy_ctrl/datasets/pick_cube_20demos \
#   --repo_id JackYuuuu/test \
#   --tag v1.0

README_TEMPLATE = """---
license: mit
task_categories:
  - robotics
tags:
  - LeRobot
  - robotic-manipulation
  - pick-and-place
  - teleoperation
configs:
  - config_name: default
    data_files: data/*/*.parquet
---

# {{ dataset_name }} Dataset

This dataset contains teleoperated demonstrations for a pick-and-place task using a {{ info.robot_type or 'robotic' }} arm with gripper.

## Dataset Description

- **Homepage**: https://huggingface.co/datasets/{{ repo_id }}
- **License**: MIT
- **Robot**: {{ info.robot_type or 'unknown' }}
- **Task**: Pick and place manipulation task
- **Collection Method**: Human teleoperation
- **FPS**: {{ info.fps or 10 }}
- **Total Episodes**: {{ info.total_episodes or 0 }}
- **Total Frames**: {{ info.total_frames or 0 }}

## Dataset Structure

### Meta Information (`meta/info.json`)

```json
{{ info_json }}
```

### Features

The dataset follows the LeRobot v{{ info.codebase_version or '2.1' }} format with the following features:

| Feature | Type | Shape | Description |
|---------|------|-------|-------------|
{%- for feature_name, feature_info in features.items() %}
| `{{ feature_name }}` | {{ feature_info.dtype }} | {{ feature_info.shape or '[]' }} | {{ feature_info.description }} |
{%- endfor %}

### Data Format

The dataset follows the LeRobot format:

```
dataset/
├── data/
│   └── chunk-000/
│       └── episode_*.parquet
├── videos/
│   └── chunk-000/
│       ├── observation.images.cam_head/
│       │   └── episode_*.mp4
│       └── observation.images.cam_wrist/
│           └── episode_*.mp4
└── meta/
    ├── info.json
    ├── episodes.jsonl
    ├── episodes_stats.jsonl
    └── tasks.jsonl
```

## Usage

Load the dataset using LeRobot:

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("{{ repo_id }}")
```

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{{ '{' }}{{ citation_key }},
    title={{ '{' }}{{ dataset_name }} Dataset{{ '}' }},
    author={{ '{' }}Dataset Creator{{ '}' }},
    year={{ '{' }}{{ current_year }}{{ '}' }},
    publisher={{ '{' }}Hugging Face{{ '}' }},
    url={{ '{' }}https://huggingface.co/datasets/{{ repo_id }}{{ '}' }}
{{ '}' }}
```

Generated on {{ generation_time }} with LeRobot dataset tools.
"""

def get_feature_description(feature_name: str, feature_info: dict) -> str:
    """Generate description for a feature based on its name and info"""
    shape = feature_info.get('shape', [])
    
    if 'joint_states' in feature_name:
        return f"Joint positions for {len(shape) if shape else 'N'}-DOF arm"
    elif 'gripper_pos' in feature_name:
        return "Gripper position"
    elif 'gripper_torque' in feature_name:
        return "Gripper torque feedback"
    elif 'images' in feature_name:
        cam_name = feature_name.split('.')[-1] if '.' in feature_name else 'camera'
        return f"{cam_name.capitalize()} camera RGB video"
    elif feature_name == 'action':
        return f"Action commands ({len(shape) if shape else 'N'} dimensions)"
    elif feature_name == 'timestamp':
        return "Timestamp in seconds"
    elif feature_name == 'episode_index':
        return "Episode identifier"
    elif feature_name == 'frame_index':
        return "Frame number within episode"
    elif feature_name == 'task':
        return "Task description"
    elif 'done' in feature_name:
        return "Episode termination flag"
    else:
        return f"{feature_name} data"

def generate_readme(data_dir: Path, repo_id: str) -> str:
    """Generate README.md content using Jinja2 template"""
    
    # Load info.json
    info_path = data_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError("meta/info.json not found")
    
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    # Prepare template variables
    dataset_name = repo_id.split('/')[-1].replace('_', ' ').title()
    citation_key = repo_id.replace('/', '_').replace('-', '_') + f"_dataset_{datetime.now().year}"
    
    # Add descriptions to features
    features = {}
    for feature_name, feature_info in info.get('features', {}).items():
        features[feature_name] = {
            **feature_info,
            'description': get_feature_description(feature_name, feature_info)
        }
    
    # Create template context
    context = {
        'dataset_name': dataset_name,
        'repo_id': repo_id,
        'info': type('Info', (), info)(),  # Convert dict to object for dot notation
        'info_json': json.dumps(info, indent=2),
        'features': features,
        'citation_key': citation_key,
        'current_year': datetime.now().year,
        'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Render template
    template = Template(README_TEMPLATE)
    return template.render(**context)

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
    
    # Generate and upload README.md
    print("Generating README.md...")
    readme_content = generate_readme(data_dir, repo_id)
    readme_path = data_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset"
    )
    print("README.md uploaded successfully")
    
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