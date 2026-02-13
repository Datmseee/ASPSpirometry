import extraction.v1_14_extract_all_cases as extraction
import model_predicting.FVLPredict as fvl_predict
import pathlib

path_to_parent = pathlib.Path(__file__).resolve().parent

EXTRACTED_PATH = pathlib.Path("extracted")
INPUT_PATH = pathlib.Path("inputs/pdfs")
GRAPHS_PATH = EXTRACTED_PATH/"graphs_png"

def do_everything():
    extraction.main(pdf_dir=path_to_parent/INPUT_PATH, out_dir=path_to_parent/EXTRACTED_PATH) # Extracts Tables & Graphs 
    labels, confidence, file_names = fvl_predict.predict_from_directory(path_to_parent/GRAPHS_PATH) # Performs predictions

    print(labels) # [[expiratory truncation, inspiratory truncation]] as np array
    print(confidence) # [[expiratory truncation confidence, inspiratory truncation confidence]] as np array
    print(file_names) # name of files as list 
    return labels, confidence, file_names

if __name__ == '__main__':
    do_everything()