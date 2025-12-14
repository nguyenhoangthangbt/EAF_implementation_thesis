"""
Feature Cache Manager with Academic Best Practices
Cites: Kuhn & Johnson (2013) - Applied Predictive Modeling (Caching for Efficiency)
       López de Prado (2018) - Advances in Financial ML (Avoiding Look-Ahead via Versioning)
"""

import os
import hashlib
import joblib
import json
from pathlib import Path
from typing import Any, Dict, Optional, List
import pandas as pd

class CacheManager_df:
    def __init__(self,df: pd.DataFrame, config: Dict[str, Any], cache_dir: str = ".././feature_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.df=df
        self.config = config
        self.metadata =  {
            'config': config,
            'shape': df.shape,
            'list_cols':"_".join(df.columns),
            'dtype':str(df.dtypes.to_dict()),
            'date_range': [str(df['Date'].min()), str(df['Date'].max())]
        }
        self.generate_cache_key()
        self.metadata_file = self.cache_dir / f"metadata_{self.cache_key}.json"
        self.cache_file = self.cache_dir / f"features_{self.cache_key}.joblib"

    def generate_cache_key(self) -> str:
        key_str = json.dumps(self.metadata, sort_keys=True, default=str)
        self.cache_key = hashlib.md5(key_str.encode()).hexdigest()#[:16]
        return self.cache_key

    def save_metadata(self):
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def load_metadata(self):
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        return self.metadata

    def save_cache_df(self):
        self.save_metadata()
        joblib.dump(self.df, self.cache_file)
        self.metadata[self.cache_key] =self.metadata
        print(f"✅ Cached features at {self.cache_file}")

    def load_cache_df(self) -> Optional[pd.DataFrame]:
        if self.cache_file.exists():
            print(f"📂 Loading cached features from {self.cache_file}")
            return joblib.load(self.cache_file)
        return None