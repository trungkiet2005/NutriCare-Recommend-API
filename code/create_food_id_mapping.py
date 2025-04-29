import pandas as pd
import json
import os

def create_food_id_mapping():
    """
    Create a JSON file that maps food IDs to their names based on the data in food_tagging.csv
    """
    print("Creating food ID to name mapping...")
    
    # Read the food data
    try:
        df = pd.read_csv('food_tagging.csv')
        print(f"Successfully loaded food dataset with {len(df)} dishes")
    except Exception as e:
        print(f"Error loading food dataset: {str(e)}")
        return
    
    # Create a dictionary mapping food IDs to names
    food_id_to_name = {}
    food_name_to_id = {}
    
    for idx, row in df.iterrows():
        food_id = idx  # Use the index as the food ID
        food_name = row['Tên món ăn'] if 'Tên món ăn' in row else None
        
        if food_name:
            food_id_to_name[str(food_id)] = food_name
            food_name_to_id[food_name] = str(food_id)
    
    # Create mapping dictionary
    mapping = {
        "id_to_name": food_id_to_name,
        "name_to_id": food_name_to_id
    }
    
    # Save to JSON file
    try:
        with open('food_id_name_mapping.json', 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=4)
        print(f"Successfully created food_id_name_mapping.json with {len(food_id_to_name)} mappings")
    except Exception as e:
        print(f"Error saving JSON file: {str(e)}")

if __name__ == "__main__":
    create_food_id_mapping()