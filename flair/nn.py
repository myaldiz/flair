import warnings
from pathlib import Path

import torch.nn

from abc import abstractmethod

from typing import Union, List

import flair
from flair.data import DataPoint
from flair.datasets import DataLoader
from flair.training_utils import Result, Metric
from flair.data import Sentence


class Model(torch.nn.Module):
    """Abstract base class for all downstream task models in Flair, such as SequenceTagger and TextClassifier.
    Every new type of model must implement these methods."""

    @abstractmethod
    def forward(
        self, data_points: Union[List[DataPoint], DataPoint]
    ) -> torch.tensor:
        """Performs a forward pass, Implement this to enable training."""
        pass

    @abstractmethod
    def calculate_loss(
            self, features: torch.tensor, sentences: List[Sentence]
    ) -> torch.tensor:
        """Calculates the loss tensor for backpropagation. Implement this to enable training."""
        pass

    @abstractmethod
    def obtain_labels(
            self, *args, **kwargs
    ):
        """Describe.."""
        pass

    @abstractmethod
    def obtain_performance_metric(self, *args, **kwargs):
        """Given inputs calculate perform"""
        pass

    @abstractmethod
    def evaluate(
        self,
        data_loader: DataLoader,
        out_path: Path = None,
        embeddings_storage_mode: str = "cpu",
    ) -> (Result, float):
        """Evaluates the model. Returns a Result object containing evaluation
        results and a loss value. Implement this to enable evaluation.
        :param data_loader: DataLoader that iterates over dataset to be evaluated
        :param out_path: Optional output path to store predictions
        :param embeddings_storage_mode: One of 'none', 'cpu' or 'gpu'. 'none' means all embeddings are deleted and
        freshly recomputed, 'cpu' means all embeddings are stored on CPU, or 'gpu' means all embeddings are stored on GPU
        :return: Returns a Tuple consisting of a Result object and a loss float value
        """
        pass

    @abstractmethod
    def _get_state_dict(self):
        """Returns the state dictionary for this model. Implementing this enables the save() and save_checkpoint()
        functionality."""
        pass

    @abstractmethod
    def _init_model_with_state_dict(state):
        """Initialize the model from a state dictionary. Implementing this enables the load() and load_checkpoint()
        functionality."""
        pass

    @abstractmethod
    def _fetch_model(model_name) -> str:
        return model_name

    def save(self, model_file: Union[str, Path]):
        """
        Saves the current model to the provided file.
        :param model_file: the model file
        """
        model_state = self._get_state_dict()

        torch.save(model_state, str(model_file), pickle_protocol=4)

    def save_checkpoint(
        self,
        model_file: Union[str, Path],
        optimizer_state: dict,
        scheduler_state: dict,
        epoch: int,
        loss: float,
    ):
        model_state = self._get_state_dict()

        # additional fields for model checkpointing
        model_state["optimizer_state_dict"] = optimizer_state
        model_state["scheduler_state_dict"] = scheduler_state
        model_state["epoch"] = epoch
        model_state["loss"] = loss

        torch.save(model_state, str(model_file), pickle_protocol=4)

    @classmethod
    def load(cls, model: Union[str, Path]):
        """
        Loads the model from the given file.
        :param model_file: the model file
        :return: the loaded text classifier model
        """
        model_file = cls._fetch_model(str(model))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # load_big_file is a workaround by https://github.com/highway11git to load models on some Mac/Windows setups
            # see https://github.com/zalandoresearch/flair/issues/351
            f = flair.file_utils.load_big_file(str(model_file))
            state = torch.load(f, map_location=flair.device)

        model = cls._init_model_with_state_dict(state)

        model.eval()
        model.to(flair.device)

        return model

    @classmethod
    def load_checkpoint(cls, checkpoint_file: Union[str, Path]):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # load_big_file is a workaround by https://github.com/highway11git to load models on some Mac/Windows setups
            # see https://github.com/zalandoresearch/flair/issues/351
            f = flair.file_utils.load_big_file(str(checkpoint_file))
            state = torch.load(f, map_location=flair.device)

        model = cls._init_model_with_state_dict(state)

        model.eval()
        model.to(flair.device)

        epoch = state["epoch"] if "epoch" in state else None
        loss = state["loss"] if "loss" in state else None
        optimizer_state_dict = (
            state["optimizer_state_dict"] if "optimizer_state_dict" in state else None
        )
        scheduler_state_dict = (
            state["scheduler_state_dict"] if "scheduler_state_dict" in state else None
        )

        return {
            "model": model,
            "epoch": epoch,
            "loss": loss,
            "optimizer_state_dict": optimizer_state_dict,
            "scheduler_state_dict": scheduler_state_dict,
        }


class LockedDropout(torch.nn.Module):
    """
    Implementation of locked (or variational) dropout. Randomly drops out entire parameters in embedding space.
    """

    def __init__(self, dropout_rate=0.5, inplace=False):
        super(LockedDropout, self).__init__()
        self.dropout_rate = dropout_rate
        self.inplace = inplace

    def forward(self, x):
        if not self.training or not self.dropout_rate:
            return x

        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout_rate)
        mask = torch.autograd.Variable(m, requires_grad=False) / (1 - self.dropout_rate)
        mask = mask.expand_as(x)
        return mask * x

    def extra_repr(self):
        inplace_str = ", inplace" if self.inplace else ""
        return "p={}{}".format(self.dropout_rate, inplace_str)


class WordDropout(torch.nn.Module):
    """
    Implementation of word dropout. Randomly drops out entire words (or characters) in embedding space.
    """

    def __init__(self, dropout_rate=0.05, inplace=False):
        super(WordDropout, self).__init__()
        self.dropout_rate = dropout_rate
        self.inplace = inplace

    def forward(self, x):
        if not self.training or not self.dropout_rate:
            return x

        m = x.data.new(x.size(0), 1, 1).bernoulli_(1 - self.dropout_rate)
        mask = torch.autograd.Variable(m, requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x

    def extra_repr(self):
        inplace_str = ", inplace" if self.inplace else ""
        return "p={}{}".format(self.dropout_rate, inplace_str)
