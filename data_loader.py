from pathlib import Path
import csv
from typing import Dict

interactions_data_path = Path(__file__).parent / 'dataset' / 'interactions.csv'

def loadInteractions():

    with open(interactions_data_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader) # Skip header

        for row in reader:



    return