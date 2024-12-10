# references:
# [1] https://github.com/DART-Laboratory/Flash-IDS
# [2] Our contribution

from dataset_prep_util import SupportedDataset, raise_unsupported_dataset

# ref. [1], [2]
def w2v_model_save_file(dataset_name: str):
    save_model_file = ""
    match dataset_name:
        case SupportedDataset.THEIA3.value:
            save_model_file = "trained_weights/theia/word2vec_theia_E3.model"
        case _:
            raise_unsupported_dataset(dataset_name)
    
    return save_model_file


