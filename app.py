from ner_task import ner_prediction as run_ner
from classification_and_relation_task import run_cls, run_rel
from data_proc import bio2typing, gen_res_for_subtask1, bio2relation, gen_res_for_subtask2
import logging
import os
import sys
from pathlib import Path
from official_eval import calculate_s1, calculate_s2
from config import (GOLD_STANDARD_1, GOLD_STANDARD_2, PRED_SUBTASK_1, PRED_SUBTASK_2,
                    RAW_TEXT_DIR, NER_OUTPUT_ROOT)


logger = logging.getLogger(__name__)


def app_run():
    # get test file ids
    test_fids = [f.stem for f in Path(RAW_TEXT_DIR).glob("*.txt")]
    logger.info("The test files are\n{}".format(test_fids))

    # run NER (5 models)
    run_ner()

    # run generating typing test data (based on each ner results)
    for i in range(5):
        bio2typing(NER_OUTPUT_ROOT.format(i), test_fids, tag=i)

    # run typing prediction
    run_cls()

    # run merging results for subtask1 (5 model ensemble)
    gen_res_for_subtask1(test_fids)

    # run generating relation test data
    bio2relation()

    # run relation prediction
    run_rel()

    # run merging results for subtask2
    gen_res_for_subtask2()

    # run evaluation
    print("Evaluation on GOLD STANDARD: ")
    print("Subtask1: ")
    calculate_s1(GOLD_STANDARD_1, PRED_SUBTASK_1, verbose=True)
    print("Subtask2: ")
    calculate_s2(GOLD_STANDARD_2, PRED_SUBTASK_2, verbose=True)


if __name__ == '__main__':
    app_run()