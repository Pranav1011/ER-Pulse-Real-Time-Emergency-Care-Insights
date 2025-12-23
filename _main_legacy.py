from src.data_processing.mimic_processor import MIMICDataProcessor
from pathlib import Path

def main():
    # Set up paths
    current_dir = Path(__file__).parent
    data_path = current_dir / 'data' / 'raw'
    output_path = current_dir / 'data' / 'processed'
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = MIMICDataProcessor(data_path=data_path)
    
    # Process data
    processed_data = processor.process_data()
    
    # Save processed data
    for name, data in processed_data.items():
        if isinstance(data, dict):
            # Handle nested dictionaries
            for sub_name, sub_data in data.items():
                # Save the processed file
                file_path = output_path / f'{name}_{sub_name}.csv'
                sub_data.to_csv(file_path, index=False)
                print(f"Saved: {file_path}")
        else:
            # Save the processed file
            file_path = output_path / f'{name}.csv'
            data.to_csv(file_path, index=False)
            print(f"Saved: {file_path}")

if __name__ == "__main__":
    main()