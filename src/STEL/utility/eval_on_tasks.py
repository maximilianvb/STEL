import inspect
import logging
import random
from typing import List, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from STEL.legacy_sim_classes import WordLengthSimilarity
from STEL.to_add_const import LOCAL_STEL_CHAR_QUAD, LOCAL_STEL_DIM_QUAD, LOCAL_ANN_STEL_DIM_QUAD
from STEL.utility.set_for_global import STYLE_TYPE_COL, MODEL_NAME_COL, ACCURACY_COL, ID_COL, CORRECT_ALTERNATIVE_COL, \
    TRIPLE, QUADRUPLE, NBR_FOR_CORRECT_COL, CLASS_THRESH, ANCHOR1_COL, ANCHOR2_COL, ALTERNATIVE11_COL, \
    ALTERNATIVE12_COL, SIMPLICITY, STYLE_DIMS, CHAR_DIMS
from STEL.utility.file_utility import read_tsv_list_to_pd, ensure_path_exists


def eval_model(stel_char_tsv: List[str] = LOCAL_STEL_CHAR_QUAD, stel_dim_tsv: List[str] = LOCAL_STEL_DIM_QUAD,
               style_objects=[WordLengthSimilarity()], output_folder='output/', eval_on_triple=False,
               only_STEL: bool = True, stel_instances: pd.DataFrame = None,
               single_predictions_save_path = None):
    """
        running the evaluation of (language) models/methods on the similarity-based STyle EvaLuation Framework (STEL)
    :param stel_char_tsv: list of paths to pandas dataframes in the expected format
    :param stel_dim_tsv:  list of paths to pandas dataframe in the expected format and majority vote column
    :param style_objects: object which can call similarities with two lists of sentences as input
    :param output_folder: where results of evaluation should be saved to ...
    :param eval_on_triple: evaluate models on the triple instead of the quadruple setup, default is False
    :param only_STEL: if the tsv file includes questions with low agreement filter those out
    :param stel_instances: pandas dataframe of pre-selected task instances, default=None
        this overwrites the tsv files stel_dim_tsv and stel_char_tsv
    :return:

    Example:
     Call all models (except for deepstyle and LIWC) on STEL:
        >>> eval_model()

     Call deepstyle extra as it has different python prerequisites (see readme)
        >>> from STEL.legacy_sim_classes import DeepstyleSimilarity
        >>> eval_model(style_objects=[DeepstyleSimilarity()])

     Call for one model only
        >>> from STEL.legacy_sim_classes import WordLengthSimilarity
        >>> eval_model(style_objects=[WordLengthSimilarity()])

     Call all models (except for deepstyle and LIWC) on the unfiltered potential task instances
          -- PROBABLY not what you want
        >>> from STEL.to_add_const import LOCAL_ANN_STEL_DIM_QUAD
        >>> eval_model(stel_dim_tsv=LOCAL_ANN_STEL_DIM_QUAD,only_STEL=False)

    """

    assert stel_dim_tsv or stel_char_tsv, 'No STEL dimension or char tsv given...'
    assert all(not inspect.isclass(style_object) for style_object in style_objects), \
        'uninstantiated classes were given as style objects... consider adding "()"?'

    logging.info("Running STEL framework ")

    if stel_instances is None:
        stel_instances, stel_types = read_in_stel_instances(stel_dim_tsv, stel_char_tsv, only_STEL)
    else:
        org_stel_types = STYLE_DIMS + CHAR_DIMS
        stel_types = org_stel_types + [stel_type for stel_type in stel_instances[STYLE_TYPE_COL].unique()
                                       if stel_type not in org_stel_types]

    # Creating the style similarity functions to evaluate on ...
    # sim_function_names, sim_object = get_simiarlity_functions(deepstyle)
    sim_function_names = [type(sim_object).__name__ for sim_object in style_objects]

    accuracy_results_df = pd.DataFrame(columns=[MODEL_NAME_COL, ACCURACY_COL] +
                                               ['Accuracy ' + style_type for style_type in stel_types])

    prediction_df = pd.DataFrame(columns=[ID_COL, CORRECT_ALTERNATIVE_COL] +
                                         [f_name for f_name in sim_function_names])

    prediction_df[ID_COL] = stel_instances[ID_COL]
    prediction_df[CORRECT_ALTERNATIVE_COL] = stel_instances[CORRECT_ALTERNATIVE_COL]
    prediction_df[STYLE_TYPE_COL] = stel_instances[STYLE_TYPE_COL]

    for f_name, f_object in zip(sim_function_names, style_objects):  # FOR every similarity function
        sim_function_callable = f_object.similarities  # getattr(sim_object, f_name)
        logging.info("Evaluation for method {}".format(f_name))

        predictions_dict = get_predictions(sim_function_callable, stel_instances, triple=eval_on_triple)

        # ACCURACY calculations
        cur_result_dict, prediction_per_instance = calculate_accuracies(stel_instances, predictions_dict,
                                                                        stel_types, f_name)

        prediction_df[f_name] = prediction_per_instance
        accuracy_results_df = pd.concat([accuracy_results_df, pd.DataFrame([cur_result_dict])], ignore_index=True)

    # LOG & SAVE results ...
    with pd.option_context('display.max_rows', None, 'display.max_columns',
                           None):  # more options can be specified also
        print(accuracy_results_df)

    if not only_STEL:
        eval_name = 'UNFILTERED'
    else:
        eval_name = 'STEL'

    model_names = ''
    if len(sim_function_names) < 5:
        for f_name in sim_function_names:
            model_names += '_' + f_name
    else:
        model_names += '_all-models'

    if eval_on_triple:
        task_setup = TRIPLE
    else:
        task_setup = QUADRUPLE

    save_filename = '{}-{}'.format(eval_name, task_setup)
    save_filename += model_names + '.tsv'
    save_path = output_folder + save_filename
    ensure_path_exists(save_path)
    accuracy_results_df.to_csv(save_path, sep='\t')
    logging.info('Saved results to {}'.format(save_path))

    if not single_predictions_save_path:
        save_filename = '{}_single-pred-{}'.format(eval_name, task_setup)
        save_filename += model_names + '.tsv'
        save_path = output_folder + save_filename
    else:
        save_path = single_predictions_save_path
    ensure_path_exists(save_path)
    prediction_df.to_csv(save_path, sep='\t')
    logging.info('Saved single predictions to {}'.format(save_path))

    return {
        "accuracies": accuracy_results_df,
        "stel_tasks": stel_instances,
        "single_predictions": prediction_df
    }


def read_in_stel_instances(stel_dim_tsv: List[str], stel_char_tsv: List[str], filter_majority_votes: bool):
    """
        read in task instances to evalaute from list of paths to tsv files
            for stel dimensions and characteristics respectively
    :param filter_majority_votes:
    :param stel_char_tsv:
    :param stel_dim_tsv:
    :return:
    """
    # read in style dimension task instances to dataframe
    if stel_dim_tsv:
        dim_instances_df = read_tsv_list_to_pd(stel_dim_tsv)
        if filter_majority_votes:
            logging.info('Filtering out tasks with low agreement ... ')
            dim_instances_df = dim_instances_df[dim_instances_df[NBR_FOR_CORRECT_COL] >= CLASS_THRESH]
        # style_dim_types = STYLE_DIMS
        stel_dim_types = list(dim_instances_df[STYLE_TYPE_COL].unique())
        stel_dim_types.reverse()
        logging.info("      on dimensions {} using files {}...".format(stel_dim_types, stel_dim_tsv))
    # read in character dimension task instances to dataframe
    if stel_char_tsv:
        char_instances_df = read_tsv_list_to_pd(stel_char_tsv)
        stel_char_types = list(char_instances_df[STYLE_TYPE_COL].unique())
        logging.info("      on characteristics {} using file {}".format(stel_char_types, stel_char_tsv))

    if stel_dim_tsv and stel_char_tsv:
        stel_instances = pd.concat([dim_instances_df, char_instances_df], ignore_index=True)
        stel_types = stel_dim_types + stel_char_types  # list(stel_instances[STYLE_TYPE_COL].unique())
        logging.info('Evaluating on {} style dim and {} style char tasks ... '
                     .format(len(dim_instances_df), len(char_instances_df)))
    elif stel_dim_tsv:
        stel_instances = dim_instances_df
        stel_types = stel_dim_types
        logging.info('Evaluating on {} style dim tasks ... '
                     .format(len(dim_instances_df)))
    else:
        stel_instances = char_instances_df
        stel_types = stel_char_types
        logging.info('Evaluating on {} style char tasks ... '
                     .format(len(char_instances_df)))

    return stel_instances, stel_types


def calculate_accuracies(stel_instances, predictions_dict, stel_types, f_name):
    """
        given the stel task instances and the predictions for the style similarity function of name f_name,
        calculate the total accuracy and the accuracies per dimension/characteristic
    :param stel_instances: dataframe with task instances
    :param predictions_dict: dictionary including 'is_random', 'nbr_random' and 'predictions'
    :param stel_dim_types:
    :param stel_char_types:
    :param f_name: name of the considred function
    :return:
    """
    pred_is_random = predictions_dict['is_random']
    nbr_random = predictions_dict['nbr_random']
    predictions = predictions_dict['predictions']
    ground_truth = stel_instances[CORRECT_ALTERNATIVE_COL].values

    # save predictions to dataframe, where prediction is set to 0 where decided randomly
    prediction_per_instance = [pred if not rnd else 0 for pred, rnd in zip(predictions, pred_is_random)]
    cur_predictions = [pred for pred, rnd in zip(predictions, pred_is_random) if not rnd]
    cur_ground_truth = [gt for gt, rnd in zip(ground_truth, pred_is_random) if not rnd]
    accuracy = get_rnd_adapted_accuracy(cur_ground_truth, cur_predictions, nbr_random)
    logging.info('  Accuracy at {}, without random {} with {} questions'
                 .format(accuracy, accuracy_score(cur_ground_truth, cur_predictions), nbr_random))
    cur_result_dict = {MODEL_NAME_COL: f_name, ACCURACY_COL: accuracy}
    stel_types_df = stel_instances[STYLE_TYPE_COL].values

    #   ACCURACY per STEL component
    for cur_stel_type in stel_types:
        nbr_tasks = sum(1 for style_type in stel_types_df if style_type == cur_stel_type)
        cur_style_pred = [pred for pred, style_type, rnd in
                          zip(predictions, stel_types_df, pred_is_random)
                          if not rnd and style_type == cur_stel_type]
        cur_style_gt = [gt for gt, style_type, rnd in
                        zip(ground_truth, stel_types_df, pred_is_random)
                        if not rnd and style_type == cur_stel_type]
        cur_nbr_random = len([1 for style_type, rnd in zip(stel_types_df, pred_is_random)
                              if style_type == cur_stel_type and rnd])
        cur_style_acc = get_rnd_adapted_accuracy(cur_style_gt, cur_style_pred, cur_nbr_random)
        logging.info('  Accuracy {} at {} for {} task instances, without random {} with {} left questions'
                     .format(cur_stel_type, cur_style_acc, nbr_tasks, accuracy_score(cur_style_gt, cur_style_pred),
                             nbr_tasks - cur_nbr_random))
        cur_result_dict['Accuracy ' + cur_stel_type] = cur_style_acc
    return cur_result_dict, prediction_per_instance


def get_rnd_adapted_accuracy(cur_ground_truth: List[int], cur_predictions: List[int], nbr_random: int) -> float:
    """
    calculate the 'random' adapted accuracy, i.e., for the number of random guesses accuracy is 0.5
    :param cur_ground_truth: list of ground truth values (ints)
    :param cur_predictions: list of predictions (ints)
    :param nbr_random: nbr of random guesses
    :return:
    """
    assert len(cur_ground_truth) == len(cur_predictions)
    if nbr_random < len(cur_predictions):
        accuracy = len(cur_predictions) / (nbr_random + len(cur_predictions)) * \
                   accuracy_score(cur_ground_truth, cur_predictions) + \
                   nbr_random / (nbr_random + len(cur_predictions)) * 0.5
    else:
        accuracy = 0.5
    return accuracy


def get_predictions(sim_function_callable: Callable, df_questions: pd.DataFrame, triple: bool = True):
    """

    :param sim_function_callable: similarity function that takes two arrays of the same length as input and
        returns an array of similairties between the respective elements at the same position in the input arrays
    :param df_questions: dataframe of the STEL questions to look at with the TSV columns
        ANCHOR1_COL, ANCHOR2_COL, ALTERNATIVE11_COL, ALTERNATIVE12_COL
    :param triple: whether to use the information of ANCHOR2_COL or not
    :return: result dict
        predictions per df_question (array of 1/2)
        is_random per df_question (array of True/False values)
        nbr_random as the number of random assignments

    """
    anchors = []
    sentences = []
    for _, question_values in df_questions.iterrows():  # iterate over rows of questions
        anchor_1 = question_values[ANCHOR1_COL]
        anchor_2 = question_values[ANCHOR2_COL]

        sentence_1 = question_values[ALTERNATIVE11_COL]
        sentence_2 = question_values[ALTERNATIVE12_COL]

        # compare anchor 1 with sentence 1
        anchors.append(anchor_1)
        sentences.append(sentence_1)
        # compare anchor 1 with sentence 2
        anchors.append(anchor_1)
        sentences.append(sentence_2)
        if not triple:
            # compare anchor 2 with sentence 1
            anchors.append(anchor_2)
            sentences.append(sentence_1)
            # compare anchor 2 with sentence 2
            anchors.append(anchor_2)
            sentences.append(sentence_2)

    sims = sim_function_callable(anchors, sentences)

    is_random, nbr_random, predictions = predict_alternatives(sims, triple)

    logging.info('random assignments: {}'.format(nbr_random))
    return {
        'predictions': predictions,
        'nbr_random': nbr_random,
        'is_random': is_random
    }


def predict_alternatives(sims, triple):
    """

    :param sims: similarity values for considered tasks (4 per question if triple=False, 2 per question if triple=True)
            following order: sim(A1,S1), sim(A1, S2), [sim(A2, S1), sim(A2, S2)]
    :param triple: whether models are evaluated on the quadruple or the triple setup
    :return:
        predictions per STEL task instance (array of 1/2),
          is a random prediction if there is no difference between the alternatives
        is_random per STEL task instance (array of True/False values)
        nbr_random as the number of random assignments
    """
    nbr_random = 0
    is_random = []
    predictions = []
    if triple:
        # predictions = [1 if sim_1 > sim_2 else 2 for sim_1, sim_2 in zip(sims[0::2], sims[1::2])]
        for sim_1, sim_2 in zip(sims[0::2], sims[1::2]):
            if sim_1 > sim_2:
                predictions.append(1)
                is_random.append(False)
            elif sim_1 < sim_2:
                predictions.append(2)
                is_random.append(False)
            else:
                random_answer = random.choice([1, 2])
                predictions.append(random_answer)
                nbr_random += 1
                is_random.append(True)
    else:
        for sim_11, sim_12, sim_21, sim_22 in zip(sims[0::4], sims[1::4], sims[2::4], sims[3::4]):
            # convert similarities in distances
            if hasattr(sim_11, 'cpu'):
                a1s1 = np.array([1 - sim_11.cpu(), 1 - sim_22.cpu()])
                a1s2 = np.array([1 - sim_12.cpu(), 1 - sim_21.cpu()])
            else:
                a1s1 = np.array([1 - sim_11, 1 - sim_22])
                a1s2 = np.array([1 - sim_12, 1 - sim_21])

            # check vector length of distances
            if np.linalg.norm(a1s1) < np.linalg.norm(a1s2):
                predictions.append(1)
                is_random.append(False)
            elif np.linalg.norm(a1s1) > np.linalg.norm(a1s2):
                predictions.append(2)
                is_random.append(False)
            else:  # np.linalg.norm(a1s1) == np.linalg.norm(a1s2)
                random_answer = random.choice([1, 2])
                predictions.append(random_answer)
                nbr_random += 1
                is_random.append(True)
    return is_random, nbr_random, predictions
