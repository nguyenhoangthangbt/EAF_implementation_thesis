#!/usr/bin/env python
import sys
import yaml
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.data_utils import fetch_yahoo_data, fetch_alpha_vantage_data
from utils.config_utils import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("data_collection")

def run_data_collection():
    """Main function to collect gold price data"""
    logger.info("Starting data collection process")
    
    # Load configuration
    config = load_config()
    data_config = config['data']
    time_config = config['time_periods']
    
    # Create data directories
    raw_data_dir = project_root / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine date range
    start_date = time_config['training']['start']
    end_date = time_config['testing']['end']
    
    logger.info(f"Fetching gold price data from {start_date} to {end_date}")
    
    # Try Yahoo Finance first
    try:
        logger.info(f"Attempting to fetch data from {data_config['source']}")
        df = fetch_yahoo_data(
            symbol=data_config['symbol'],
            start=start_date,
            end=end_date
        )
        source_used = data_config['source']
        logger.info(f"Successfully fetched data from {source_used}")
    except Exception as e:
        logger.warning(f"Failed to fetch data from {data_config['source']}: {str(e)}")
        logger.info(f"Attempting to fetch data from {data_config['backup_source']}")
        
        try:
            df = fetch_alpha_vantage_data(
                symbol=data_config['symbol'],
                start=start_date,
                end=end_date
            )
            source_used = data_config['backup_source']
            logger.info(f"Successfully fetched data from {source_used}")
        except Exception as e:
            logger.error(f"Failed to fetch data from {data_config['backup_source']}: {str(e)}")
            raise
    
    # Save raw data
    output_path = raw_data_dir / f"gold_price_{source_used}_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(output_path)
    logger.info(f"Raw data saved to {output_path}")
    
    # Save metadata
    metadata = {
        "source": source_used,
        "start_date": str(df.index.min()),
        "end_date": str(df.index.max()),
        "records_count": len(df),
        "columns": list(df.columns)
    }
    
    metadata_path = raw_data_dir / f"metadata_{datetime.now().strftime('%Y%m%d')}.yaml"
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f)
    
    logger.info("Data collection process completed successfully")
    return 0
