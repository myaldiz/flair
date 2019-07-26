import os
from pathlib import Path
from typing import Tuple, Union
from abc import abstractmethod
import logging

import torch

import ray
from ray import tune
from ray.tune.schedulers import HyperBandScheduler
from ray.tune import Trainable

import flair
from flair.data import Corpus
from flair.training_utils import EvaluationMetric
from .param_selection import ParamSelector, OptimizationValue
from .param_selection import SearchSpace

log = logging.getLogger("flair")


class TuneParamSelector(Trainable):
    def _setup(self, config):
        args = config["args"]
        config.pop("args", None)
        self.params = config

        torch.manual_seed(args["seed"])
        if args.cuda:
            torch.cuda.manual_seed(args["seed"])


    def _train_iteration(self):
        return self._set_up_model(self.params)

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
        args["cuda"] = self.use_gpu

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
