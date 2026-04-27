import os
from config import PROJECT_ROOT


def clear_folder(data_cat: str):
    
    # Choose folder to delete
    all_types = {'stock' : 'stock_price_cache', 
                 'index': "index_cache", 
                 'news': 'new_cache'}
    if data_cat not in all_types.keys():
        raise ValueError('data_cat argument only accept stock, index, news')

    # Determine folder path
    folder_path = PROJECT_ROOT / "data"/ "cached_data" / all_types[data_cat]

    # Check for path existence
    if not os.path.exists(folder_path):
        raise ValueError(f"Path does not exist: {folder_path}")
    if not os.path.isdir(folder_path):
        raise ValueError(f"Not a directory: {folder_path}")

    # ---- MAIN DELETE -----
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        try:
            os.remove(file_path)
            
        except PermissionError:
            print(f'{filename} is not a File but maybe a Folder')
            continue

        except Exception as e:
            print(f"Failed to delete {file_path}: {type(e).__name__} {e}")
            continue
    
    print('Finish clearing cache')

if __name__ == '__main__':
    data_cat = 'stock'
    clear_folder(data_cat)

# CMD: python -m utils.clear_data_cache