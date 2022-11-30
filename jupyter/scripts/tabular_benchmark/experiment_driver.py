#!/usr/bin/env python
# coding: utf-8
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License
# 2.0 and the following additional limitation. Functionality enabled by the
# files subject to the Elastic License 2.0 may only be used in production when
# invoked by an Elasticsearch process with a license key installed that permits
# use of machine learning features. You may not use this file except in
# compliance with the Elastic License 2.0 and the foregoing additional
# limitation.

import shutil
import random
import string

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from incremental_learning.config import data_dir, datasets_dir, jobs_dir, logger
from incremental_learning.elasticsearch import push2es
from incremental_learning.job import Job, evaluate, train, update

from incremental_learning.trees import Forest
from pathlib2 import Path
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn.model_selection import train_test_split

import time 
import numpy as np
from sklearn.metrics import r2_score
import pandas as pd

from incremental_learning.benchmark.generate_dataset_pipeline import generate_dataset
from incremental_learning.job import train, evaluate

experiment_name = 'tabular-benchmark-test'
uid = ''.join(random.choices(string.ascii_lowercase, k=6))
experiment_data_path = Path('/tmp/'+experiment_name)
ex = Experiment(name=uid)
ex.observers.append(FileStorageObserver(experiment_data_path))
ex.logger = logger


@ex.config
def my_config():
    train_prop = 0.70,
    val_test_prop = 0.3
    max_val_samples = 50000
    max_test_samples = 50000
    n_iter = 1 #"auto"

    force_update = True
    verbose = False
    dataset_name = 'house'
    test_fraction = 0.1
    training_fraction = 0.1
    update_fraction = 0.1
    threads = 8
    update_steps = 10
    submit = False # submit to elasticsearch cluster?

    CONFIG_DEFAULT = {
        "train_prop": 0.70,
        "val_test_prop": 0.3,
        "max_val_samples": 50000,
        "max_test_samples": 50000
    }

    analysis_parameters = {'parameters':
                           {'tree_topology_change_penalty': 0.0,
                            'prediction_change_cost': 0.0,
                            'data_summarization_fraction': 1.0,
                            'early_stopping_enabled': False,
                            'max_optimization_rounds_per_hyperparameter': 5}}
    sampling_mode = 'nlargest'
    n_largest_multiplier = 1



# @ex.capture
# def compute_error(job, dataset, dataset_name, verbose, _run):
#     job_eval = evaluate(dataset_name=dataset_name, dataset=dataset,
#                         original_job=job, run=_run, verbose=verbose)
#     job_eval.wait_to_complete()
#     dependent_variable = job.dependent_variable
#     if job.is_regression():
#         y_true = np.array([y for y in dataset[dependent_variable]])
#         return metrics.mean_squared_error(y_true, job_eval.get_predictions())
#     elif job.is_classification():
#         y_true = np.array([str(y) for y in dataset[dependent_variable]])
#         return metrics.accuracy_score(y_true,
#                                       job_eval.get_predictions())
#     else:
#         _run.run_logger.warning(
#             "Job is neither regression nor classification. No metric scores are available.")
#     return -1


# @ex.capture
# def get_residuals(job, dataset, dataset_name, verbose, _run):
#     job_eval = evaluate(dataset_name=dataset_name, dataset=dataset,
#                         original_job=job, run=_run, verbose=verbose)
#     job_eval.wait_to_complete()
#     predictions = job_eval.get_predictions()
#     residuals = np.absolute(dataset[job_eval.dependent_variable] - predictions)
#     return residuals


# def get_forest_statistics(model_definition):
#     trained_models = model_definition['trained_model']['ensemble']['trained_models']
#     forest = Forest(trained_models)
#     result = {}
#     result['num_trees'] = len(forest)
#     result['tree_nodes_max'] = forest.max_tree_length()
#     result['tree_nodes_mean'] = np.mean(forest.tree_sizes())
#     result['tree_nodes_std'] = np.std(forest.tree_sizes())
#     result['tree_depth_mean'] = np.mean(forest.tree_depths())
#     result['tree_depth_std'] = np.std(forest.tree_depths())
#     return result


# def remove_subset(dataset, subset):
#     dataset = dataset.merge(
#         subset, how='outer', indicator=True).loc[lambda x: x['_merge'] == 'left_only']
#     dataset.drop(columns=['_merge'], inplace=True)
#     return dataset


# @ex.capture
# def sample_points(total_dataset, update_num_samples, used_dataset, random_state, _run):
#     D_update = None
#     success = False
#     k = 2
#     while not success:
#         if update_num_samples*k <= total_dataset.shape[0]:
#             D_update = None
#             break
#         largest = total_dataset.sample(
#             n=update_num_samples*k, weights='indicator', random_state=random_state)

#         largest.drop(columns=['indicator'], inplace=True)
#         D_update = remove_subset(largest, used_dataset)
#         if D_update.shape[0] < update_num_samples:
#             k += 1
#             continue
#         if D_update.shape[0] >= update_num_samples:
#             D_update = D_update.sample(n=update_num_samples)
#             success = True
#     _run.run_logger.info("Sampling completed after with k={}.".format(k))
#     return D_update

@ex.capture
def update_config(data_transform_config: dict, benchmark: dict) -> dict:
     # Use the appropriate model config
    model_config = { }
    dataset_size=benchmark["dataset_size"]
    categorical=benchmark["categorical"]
    regression=benchmark["task"] == "regression"

    if dataset_size == "medium":
        data_transform_config["max_train_samples"] = 10000
    elif dataset_size == "large":
        data_transform_config["max_train_samples"] = 50000


    if categorical:
        data_transform_config["data__categorical"] =  True
    else:
        data_transform_config["data__categorical"] = False

    if regression:
        data_transform_config["regression"] = True
        data_transform_config["data__regression"] = True
    else:
        data_transform_config["regression"] = False
        data_transform_config["data__regression"] = False

    data_transform_config["data__keyword"] = benchmark['datasets']

    config = {
        "program": "run_experiment.py",
        "metric": {
            "name": "mean_test_score",
            "goal": "minimize"  # RMSE
        } if regression else {
            "name": "mean_test_score",
            "goal": "maximize"  # accuracy
        },
        "parameters": dict(model_config, **data_transform_config)
    }
    return config


@ex.main
def my_main(_run, _seed, n_iter, CONFIG_DEFAULT, verbose):
    if 'comment' not in _run.meta_info or not _run.meta_info['comment']:
        raise RuntimeError("Specify --comment parameter for this experiment.")
    results = {}
    data_transform_config = {"data__method_name": "real_data"}
    benchmarks = [
    {
        "categorical": False,
        "task": "regression",
        "dataset_size": "medium",
        "datasets": "house_sales",
    }
    ]
    benchmark = benchmarks[0]

    config = update_config(
    data_transform_config=data_transform_config, benchmark=benchmark
    )
    config["parameters"] = {**config["parameters"], **CONFIG_DEFAULT}
    _run.run_logger.info(config)
    train_scores = []
    val_scores = []
    test_scores = []
    r2_train_scores = []
    r2_val_scores = []
    r2_test_scores = []
    times = []
    if n_iter == "auto":
        (
            x_train,
            x_val,
            x_test,
            y_train,
            y_val,
            y_test,
            categorical_indicator,
        ) = generate_dataset(config["parameters"], np.random.RandomState(0))
        if x_test.shape[0] > 6000:
            n_iter = 1
        elif x_test.shape[0] > 3000:
            n_iter = 2
        elif x_test.shape[0] > 1000:
            n_iter = 3
        else:
            n_iter = 5
    for i in range(n_iter):
        # if config["log_training"]: #FIXME
        #    config["model__wandb_run"] = run
        rng = np.random.RandomState(i)
        print(rng.randn(1))
        # TODO: separate numeric and categorical features
        t = time.time()
        (
            x_train,
            x_val,
            x_test,
            y_train,
            y_val,
            y_test,
            categorical_indicator,
        ) = generate_dataset(config['parameters'], rng)
        data_generation_time = time.time() - t
        _run.run_logger.info(f"Data generation time: {data_generation_time}")
        print(x_train.shape)
        x_train_df = pd.DataFrame(x_train, columns=['f' + str(col_idx) for col_idx in range(x_train.shape[1])])
        categorical_fields = [field for field, indicator in zip(x_train_df.columns, categorical_indicator) if indicator]
        x_train_df['target'] = y_train
        train_config = {
            'job_id': benchmark['datasets'],
            'rows': (x_train.shape[0] + x_val.shape[0]),
            'cols': x_train.shape[1]+1,
            'memory_limit': 50000000,
            'threads': 8,
            'results_field': 'ml',
            'categorical_fields': categorical_fields,
            'analysis': {
                'name': benchmark['task'],
                'parameters': {
                    'randomize_seed': rng.randint(100000000),
                    'dependent_variable': 'target'
                }
            }
        }
        job = train(train_config['job_id'], x_train_df, config=train_config, verbose=verbose)
        elapsed_time = job.wait_to_complete()
        _run.run_logger.info(f"Training completed after {elapsed_time} seconds.")

        predictions = job.get_predictions()
        _run.run_logger.info("R2 score train: {}".format(r2_score(predictions, y_train)))

        evaluate_job = evaluate(dataset_name=train_config['job_id'], dataset=x_train_df, original_job=job, config=train_config, verbose=verbose)
        elapsed_time = evaluate_job.wait_to_complete()
        _run.run_logger.info(f"Evaluation completed after {elapsed_time} seconds.")
        _run.run_logger.info("R2 score evaluate: {}".format(r2_score(evaluate_job.get_predictions(), y_train)))

    # _run.config['analysis'] = analysis_parameters

    # random_state = np.random.RandomState(seed=_seed)

    # baseline_model_name = "e{}_s{}_d{}_t{}".format(
    #     experiment_name, _seed, dataset_name, training_fraction)

    # original_dataset = read_dataset()
    # train_dataset, test_dataset = train_test_split(
    #     original_dataset, test_size=test_fraction, random_state=random_state)
    # train_dataset = train_dataset.copy()
    # test_dataset = test_dataset.copy()
    # baseline_dataset = train_dataset.sample(
    #     frac=training_fraction, random_state=random_state)
    # update_num_samples = int(train_dataset.shape[0]*update_fraction)

    # _run.run_logger.info(
    #     "Baseline training started for {}".format(baseline_model_name))
    # if job_exists(baseline_model_name, remote=True):
    #     path = download_job(baseline_model_name)
    #     baseline_model = Job.from_file(path)
    # else:
    #     baseline_model = train(dataset_name, baseline_dataset,
    #                            verbose=verbose, run=_run)
    #     elapsed_time = baseline_model.wait_to_complete()
    #     path = jobs_dir / baseline_model_name
    #     baseline_model.store(path)
    #     upload_job(path)
    # results['baseline'] = {}
    # results['baseline']['hyperparameters'] = baseline_model.get_hyperparameters()
    # results['baseline']['forest_statistics'] = get_forest_statistics(
    #     baseline_model.get_model_definition())
    # results['baseline']['train_error'] = compute_error(
    #     baseline_model, train_dataset)
    # results['baseline']['test_error'] = compute_error(
    #     baseline_model, test_dataset)
    # _run.run_logger.info("Baseline training completed")
    # used_dataset = baseline_dataset.copy()

    # previous_model = baseline_model
    # for step in range(update_steps):
    #     _run.run_logger.info("Update step {} started".format(step))

    #     _run.run_logger.info("Sampling started")
    #     if sampling_mode == 'nlargest':
    #         train_dataset['indicator'] = get_residuals(
    #             previous_model, train_dataset)

    #         D_update = sample_points(
    #             train_dataset, update_num_samples, used_dataset, random_state)
    #         train_dataset.drop(columns=['indicator'], inplace=True)
    #     else:
    #         D_update = train_dataset.sample(
    #             n=update_num_samples, random_state=random_state)
    #     _run.run_logger.info("Sampling completed")
    #     if D_update is None:
    #         _run.run_logger.warning(
    #             "Update loop interrupted. It seems that you used up all the data!")
    #         break

    #     _run.run_logger.info("Update started")
    #     hyperparameters = previous_model.get_hyperparameters()
    #     for name, value in hyperparameters.items():
    #         if name in ['soft_tree_depth_limit']:
    #             min, max = (value, value*1.5)
    #             hyperparameters[name] = [min, max]
    #         elif name == 'retrained_tree_eta':
    #             hyperparameters[name] = (0.1, 0.6)

    #     updated_model = update(dataset_name=dataset_name, dataset=D_update, original_job=previous_model,
    #                            force=force_update, hyperparameter_overrides=hyperparameters, verbose=verbose, run=_run)
    #     elapsed_time = updated_model.wait_to_complete(clean=False)
    #     _run.log_scalar('updated_model.elapsed_time', elapsed_time)
    #     _run.log_scalar('updated_model.train_error',
    #                     compute_error(updated_model, train_dataset))
    #     _run.log_scalar('updated_model.test_error',
    #                     compute_error(updated_model, test_dataset))
    #     _run.log_scalar('training_fraction', training_fraction)
    #     _run.log_scalar('seed', _seed)
    #     used_dataset = pd.concat([used_dataset, D_update])
    #     _run.run_logger.info(
    #         "New size of used data is {}".format(used_dataset.shape[0]))

    #     _run.log_scalar('updated_model.hyperparameters',
    #                     updated_model.get_hyperparameters())
    #     _run.log_scalar('run.comment', _run.meta_info['comment'])
    #     _run.log_scalar('run.config', _run.config)
    #     for k, v in get_forest_statistics(updated_model.get_model_definition()).items():
    #         _run.log_scalar('updated_model.forest_statistics.{}'.format(k), v)
    #     updated_model.clean()
    #     _run.run_logger.info("Update completed")

    #     previous_model = updated_model
    #     _run.run_logger.info("Update step completed".format(step))

    return results


if __name__ == '__main__':
    run = ex.run_commandline()
    run_data_path = experiment_data_path / str(run._id)
    import incremental_learning.job
    ex.add_artifact(incremental_learning.job.__file__)
    if run.config['submit']:
        push2es(data_path=run_data_path, name=experiment_name)
    else:
        run.run_logger.info(
            "Experiment results were not submitted to the index.")
    shutil.rmtree(run_data_path)
