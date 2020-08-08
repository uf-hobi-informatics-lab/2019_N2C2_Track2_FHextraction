from ClinicalTransformerNER.src.run_transformer_batch_prediction import main as ner
from config import NER_MODELS, RAW_TEXT_DIR, BIO_TEXT_DIR, NER_OUTPUT_ROOT
from pathlib import Path


class Args:
    def __init__(self, pretrained_model=None, raw_text_dir=None,
                 preprocessed_text_dir=None, output_dir=None):
        self.pretrained_model = pretrained_model
        self.model_type = "bert"
        self.preprocessed_text_dir = preprocessed_text_dir
        self.raw_text_dir = raw_text_dir
        self.data_has_offset_information = True
        self.output_dir = output_dir
        self.do_lower_case = True
        self.eval_batch_size = 8
        self.max_seq_length = 128
        self.log_file = None
        self.log_lvl = "i"
        self.do_format = 0
        self.do_copy = False
        self.progress_bar = False
        self.use_crf = False


def ner_prediction():
    args = Args()
    args.raw_text_dir = RAW_TEXT_DIR
    args.preprocessed_text_dir = BIO_TEXT_DIR
    for ner_model in NER_MODELS:
        model_path = Path(ner_model)  # #./models/2019_n2c2_fh_ner_0
        mid = ner_model.name.split("_")[-1]
        args.pretrained_model = model_path
        args.output_dir = NER_OUTPUT_ROOT.format(mid)
        # Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        ner(args)
