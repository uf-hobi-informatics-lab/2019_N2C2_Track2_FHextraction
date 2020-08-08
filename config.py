#################################
# Configuration file
#################################

"""
all the models used for prediction
You have to download the models using the URLs in the readme and put them under the ./models directory
"""
MODEL_TYPE = "bert-large-uncased-whole-word-masking"

NER_MODELS = ["./models/2019_n2c2_fh_ner_0", "./models/2019_n2c2_fh_ner_1", "./models/2019_n2c2_fh_ner_2",
              "./models/2019_n2c2_fh_ner_3", "./models/2019_n2c2_fh_ner_4"]
NEGATION_MODEL = "./models/2019_n2c2_fh_obn/production"
FMROLE_MODEL = "./models/2019_n2c2_fh_fmr/production"
FMSIDE_MODEL = "./models/2019_n2c2_fh_fms/production"
LS_MODEL = "./models/2019_n2c2_fh_lss/production"
RELATION_MODEL = "./models/2019_n2c2_fh_rel"

RAW_TEXT_DIR = "../sample_data/raw_text"
PREPROCESSED_TEXT_DIR = "../sample_data/preprocessed_text"
TEXT_AS_SENT_DIR = "../sample_data/text_as_sents"
BIO_TEXT_DIR = "../sample_data/bio"
NER_OUTPUT_ROOT = "../results/pred/ner_{}"

CLS_OUTPUT_ROOT = "./results/pred/{}_{}"
NER_TYPING_ROOT = "./sample_data/cls"
OBN_TEST = "./sample_data/cls/obn_{}"
FMR_TEST = "./sample_data/cls/fmrs_{}"
FMS_TEST = "./sample_data/cls/fmrs_{}"
LSS_TEST = "./sample_data/cls/lss_{}"
REL_OUTPUT_ROOT = "./results/pred/rel"
REL_TEST = "./sample_data/rel"

ENSEMBLE_THRESHOLD = 3
GLOBAL_CUTOFF = 2

GOLD_STANDARD_1 = "./results/gs/test_subtask1_1055.tsv"
GOLD_STANDARD_2 = "./results/gs/test_subtask2_1055.tsv"

PRED_SUBTASK_1 = "./results/pred/test_subtask1_1055.tsv"
PRED_SUBTASK_2 = "./results/pred/test_subtask2_1055.tsv"
