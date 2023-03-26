import pandas as pd
import os


def load_data(file_table_pth: str) -> pd.DataFrame:
    """Given a table containing file names 
    returns compiled dataframe containing text bodies 
    and abstract bodies"""
    if not os.path.exists(file_table_pth):
        raise Exception(f"File path {file_table_pth} does not exist.")
    
    paths = pd.read_csv(file_table_pth)
    texts = pd.DataFrame({'id': [], 'body': [], 'abstract': []})

    for row in paths.itertuples():
        try:
            with open(f"/Data_Fetching/articles/{row['raw_path']}") as b:
                body = b
            with open(f"/Data_Fetching/abstracts/{row['abstract_path']}") as abstract:
                df = pd.DataFrame({'id': [row['body'][:-6]], 'body': [body.read()], 'abstract': [abstract.read()]})
                texts = pd.concat([texts,df])

        except:
            raise Exception(f"File path(s) for ID {row['body'][:-6]} not found, or error in opening...")

    return texts

def save_dataset(data: pd.Dataframe, name: str) -> str:
    """
    Arguments:
        Data: Pandas Dataframe that contains data {id, body, abstract}
    Returns:
        String of compiled CSV path
    """





        