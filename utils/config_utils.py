import sys
import yaml
from pathlib import Path
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
def load_config(config_filename="config.yaml"):
    """Load configuration from YAML file"""
    config_path = project_root / "config" / config_filename
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config