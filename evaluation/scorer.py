import sys
import argparse
import logging.handlers
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import evaluation.src.annotation as an
import evaluation.src.annotations as ans
import evaluation.src.propaganda_techniques as pt

logger = logging.getLogger("propaganda_scorer")
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.setLevel(logging.INFO)


def main(user_submission_file, gold_file, output_log_file, propaganda_techniques_list_file):


    output_for_script = True

    if not output_for_script:
        logger.addHandler(ch)

    # if args.debug_on_std:
    #     ch.setLevel(logging.DEBUG)

    if output_log_file is not None:
        logger.info("Logging execution to file " + output_log_file)
        fileLogger = logging.FileHandler(output_log_file)
        fileLogger.setLevel(logging.DEBUG)
        fileLogger.setFormatter(formatter)
        logger.addHandler(fileLogger)

    propaganda_techniques = pt.Propaganda_Techniques(propaganda_techniques_list_file)
    an.Annotation.set_propaganda_technique_list_obj(propaganda_techniques)

    user_annotations = ans.Annotations()
    user_annotations.load_annotation_list_from_file(user_submission_file)
    for article in user_annotations.get_article_id_list():
        user_annotations.get_article_annotations_obj(article).sort_spans()

    gold_annotations = ans.Annotations()
    gold_annotations.load_annotation_list_from_file(gold_file)
    for article in gold_annotations.get_article_id_list():
        gold_annotations.get_article_annotations_obj(article).sort_spans()

    logger.info("Checking format: User Predictions -- Gold Annotations")
    if not user_annotations.compare_annotations_identical_article_lists(gold_annotations) or not user_annotations.compare_annotations_identical(gold_annotations):
        logger.error("wrong format, no scoring will be performed")
        sys.exit()
    logger.info("OK: submission file format appears to be correct")
    # change - anni
    res_for_output, res_for_script, f1, precision, recall, f1_per_class, precision_per_class, recall_per_class = user_annotations.TC_score_to_string(gold_annotations, output_for_script)
    logger.info("Scoring submission" + res_for_output)
    if output_for_script:
        print(res_for_script)
    # change - anni
    return f1, precision, recall, f1_per_class, precision_per_class, recall_per_class