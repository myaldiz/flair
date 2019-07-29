import logging
from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import Tuple, Union
import types
import numpy as np

import torch

import ray
from ray import tune
from ray.tune.schedulers import HyperBandScheduler
from ray.tune import Trainable

from hyperopt import hp, fmin, tpe

import flair.nn
from flair.data import Corpus
from flair.embeddings import DocumentPoolEmbeddings, DocumentRNNEmbeddings
from flair.hyperparameter import Parameter
from flair.hyperparameter.parameter import (
    SEQUENCE_TAGGER_PARAMETERS,
    TRAINING_PARAMETERS,
    DOCUMENT_EMBEDDING_PARAMETERS,
    MODEL_TRAINER_PARAMETERS,
)
from flair.models import SequenceTagger, TextClassifier
from flair.trainers import ModelTrainer
from flair.training_utils import (
    EvaluationMetric,
    log_line,
    init_output_file,
    WeightExtractor,
    add_file_handler,
)

log = logging.getLogger("flair")


class OptimizationValue(Enum):
    DEV_LOSS = "loss"
    DEV_SCORE = "score"


class SearchSpace(object):
    def __init__(self):
        self.search_space = {}

    def add(self, parameter: Parameter, func, **kwargs):
        self.search_space[parameter.value] = func(parameter.value, **kwargs)

    def get_search_space(self):
        return hp.choice("parameters", [self.search_space])


class SequentialParamSelector(object):
    def __init__(
        self,
        corpus: Corpus,
        base_path: Union[str, Path],
        max_epochs: int,
        evaluation_metric: EvaluationMetric,
        training_runs: int,
        optimization_value: OptimizationValue,
    ):
        if type(base_path) is str:
            base_path = Path(base_path)

        self.corpus = corpus
        self.max_epochs = max_epochs
        self.base_path = base_path
        self.evaluation_metric = evaluation_metric
        self.run = 1
        self.training_runs = training_runs
        self.optimization_value = optimization_value

        self.param_selection_file = init_output_file(base_path, "param_selection.txt")

    @abstractmethod
    def _set_up_model(self, params: dict) -> flair.nn.Model:
        pass

    def _objective(self, params: dict):
        log_line(log)
        log.info(f"Evaluation run: {self.run}")
        log.info(f"Evaluating parameter combination:")
        for k, v in params.items():
            if isinstance(v, Tuple):
                v = ",".join([str(x) for x in v])
            log.info(f"\t{k}: {str(v)}")
        log_line(log)

        for sent in self.corpus.get_all_sentences():
            sent.clear_embeddings()

        scores = []
        vars = []

        for i in range(0, self.training_runs):
            log_line(log)
            log.info(f"Training run: {i + 1}")

            model = self._set_up_model(params)

            training_params = {
                key: params[key] for key in params if key in TRAINING_PARAMETERS
            }
            model_trainer_parameters = {
                key: params[key] for key in params if key in MODEL_TRAINER_PARAMETERS
            }

            trainer: ModelTrainer = ModelTrainer(
                model, self.corpus, **model_trainer_parameters
            )

            result = trainer.train(
                self.base_path,
                max_epochs=self.max_epochs,
                param_selection_mode=True,
                **training_params,
            )

            # take the average over the last three scores of training
            if self.optimization_value == OptimizationValue.DEV_LOSS:
                curr_scores = result["dev_loss_history"][-3:]
            else:
                curr_scores = list(
                    map(lambda s: 1 - s, result["dev_score_history"][-3:])
                )

            score = sum(curr_scores) / float(len(curr_scores))
            var = np.var(curr_scores)
            scores.append(score)
            vars.append(var)

        # take average over the scores from the different training runs
        final_score = sum(scores) / float(len(scores))
        final_var = sum(vars) / float(len(vars))

        test_score = result["test_score"]
        log_line(log)
        log.info(f"Done evaluating parameter combination:")
        for k, v in params.items():
            if isinstance(v, Tuple):
                v = ",".join([str(x) for x in v])
            log.info(f"\t{k}: {v}")
        log.info(f"{self.optimization_value.value}: {final_score}")
        log.info(f"variance: {final_var}")
        log.info(f"test_score: {test_score}\n")
        log_line(log)

        with open(self.param_selection_file, "a") as f:
            f.write(f"evaluation run {self.run}\n")
            for k, v in params.items():
                if isinstance(v, Tuple):
                    v = ",".join([str(x) for x in v])
                f.write(f"\t{k}: {str(v)}\n")
            f.write(f"{self.optimization_value.value}: {final_score}\n")
            f.write(f"variance: {final_var}\n")
            f.write(f"test_score: {test_score}\n")
            f.write("-" * 100 + "\n")

        self.run += 1

        return {"status": "ok", "loss": final_score, "loss_variance": final_var}

    def optimize(self, space: SearchSpace, max_evals=100):
        search_space = space.search_space
        best = fmin(
            self._objective, search_space, algo=tpe.suggest, max_evals=max_evals
        )

        log_line(log)
        log.info("Optimizing parameter configuration done.")
        log.info("Best parameter configuration found:")
        for k, v in best.items():
            log.info(f"\t{k}: {v}")
        log_line(log)

        with open(self.param_selection_file, "a") as f:
            f.write("best parameter combination\n")
            for k, v in best.items():
                if isinstance(v, Tuple):
                    v = ",".join([str(x) for x in v])
                f.write(f"\t{k}: {str(v)}\n")


class TuneParamSelector(Trainable):
    def _setup(self, config):
        args = config["args"]
        config.pop("args", None)
        self.params = config
        self.corpus = args["corpus"]

        torch.manual_seed(args["seed"])
        if args.cuda:
            torch.cuda.manual_seed(args["seed"])

        model = args["_set_up_model"](self.params)

        log_line(log)
        log.info(f'Model: "{self.model}"')
        log_line(log)
        log.info(f'Corpus: "{self.corpus}"')
        log_line(log)
        log.info("Parameters:")
        log.info(f' - learning_rate: "{config["learning_rate"]}"')
        # log.info(f' - mini_batch_size: "{config["mini_batch_size"]}"')
        # log.info(f' - patience: "{config["patience"]}"')
        # log.info(f' - anneal_factor: "{config["anneal_factor"]}"')

        # Check this once more
        weight_extractor = None  # WeightExtractor()

        training_params = {
            key: config[key] for key in config if key in TRAINING_PARAMETERS
        }

        model_trainer_parameters = {
            key: config[key] for key in config if key in MODEL_TRAINER_PARAMETERS
        }

        # This should be enough for initializing all the parameters for the trainer
        self.trainer = ModelTrainer(model, self.corpus, model_trainer_parameters)

    def _train_iteration(self):
        trainer = self.trainer
        trainer.train_epoch(
            trainer.epoch,
        )


    def _test(self):
        pass

    def _train(self):
        self._train_iteration()
        return self._test()

    def _save(self, checkpoint_dir):
        pass

    def _restore(self, checkpoint_path):
        pass


class DistributedParamSelector(object):
    def __init__(
            self,
            redis_address: str,
            corpus: Corpus,
            base_path: Union[str, Path],
            max_epochs: int,
            evaluation_metric: EvaluationMetric,
            training_runs: int,
            optimization_value: OptimizationValue,
            use_gpu=torch.cuda.is_available(),
    ):
        self.corpus = corpus
        self.max_epochs = max_epochs
        self.base_path = base_path
        self.evaluation_metric = evaluation_metric
        self.training_runs = training_runs
        self.optimization_value = optimization_value
        self.use_gpu = use_gpu

        ray.init(redis_address=redis_address)
        self.hb_scheduler = HyperBandScheduler(
            time_attr="training_iteration",
            metric="mean_loss", mode="min"
        )

        # Config dictionary for Tune Param Selector
        config, args = dict(), dict()
        self.config = config
        config["args"] = args
        args["cuda"] = self.use_gpu
        args["corpus"] = corpus

    def optimize(
            self,
            space: SearchSpace,
            max_evals=100,
            random_seed=1
    ):
        config = self.config
        args = config["args"]
        config.update(space.search_space)
        args["max_evals"] = max_evals
        args["seed"] = random_seed
        args["_set_up_model"] = self._set_up_model

        tune.run(
            TuneParamSelector,
            scheduler=self.hb_scheduler,
            **{
                "stop": {
                    "training_iteration": self.max_epochs,
                },
                "resources_per_trial": {
                    "cpu": 3,
                    "gpu": 1 if self.use_gpu else 0,
                },
                "num_samples": 20,
                "checkpoint_at_end": True,
                "config": config
            })

    @abstractmethod
    def _set_up_model(self, params: dict) -> flair.nn.Model:
        pass


def ParamSelector(
        corpus: Corpus,
        tag_type: str,
        base_path: Union[str, Path],
        max_epochs: int = 50,
        distributed=False,
        model_type=SequenceTagger,
        multi_label: bool = True,
        document_embedding_type=None,
        evaluation_metric: EvaluationMetric = EvaluationMetric.MICRO_F1_SCORE,
        training_runs: int = 1,
        optimization_value: OptimizationValue = OptimizationValue.DEV_LOSS,
):
    param_selector = None
    param_selector_class = DistributedParamSelector if distributed else SequentialParamSelector
    param_selector = param_selector_class(
        corpus, base_path,
        evaluation_metric,
        training_runs,
        optimization_value
    )
    param_selector.model_type = model_type

    def _set_up_model(self, params: dict):

        if self.model_type == SequenceTagger:
            parameter_set = SEQUENCE_TAGGER_PARAMETERS
        elif self.model_type == TextClassifier:
            parameter_set = DOCUMENT_EMBEDDING_PARAMETERS
        model_params = {
            key: params[key] for key in params if key in parameter_set
        }

        if self.model_type == SequenceTagger:
            self.tag_type = tag_type
            self.tag_dictionary = self.corpus.make_tag_dictionary(self.tag_type)

            model: SequenceTagger = SequenceTagger(
                tag_dictionary=self.tag_dictionary,
                tag_type=self.tag_type,
                **model_params,
            )
        elif self.model_type == TextClassifier:
            self.multi_label = multi_label
            self.document_embedding_type = document_embedding_type

            self.label_dictionary = self.corpus.make_label_dictionary()

            if self.document_embedding_type == "lstm":
                document_embedding = DocumentRNNEmbeddings(**model_params)
            else:
                document_embedding = DocumentPoolEmbeddings(**model_params)

            model: TextClassifier = TextClassifier(
                label_dictionary=self.label_dictionary,
                multi_label=self.multi_label,
                document_embeddings=document_embedding,
            )
        else:
            log.error("Unknown class type for parameter selection")
            raise TypeError

        # We bind _set_up_model method to the appropriate ParamSelector class
        # specified by model_type variable
        param_selector._set_up_model = types.MethodType(_set_up_model, param_selector)
        return param_selector
