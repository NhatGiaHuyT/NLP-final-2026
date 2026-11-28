"""Dataset interface so model / evaluation code does not depend on HousingQA."""

from abc import ABC, abstractmethod
from typing import Iterator

import pandas as pd


class LegalQADataset(ABC):
    name: str

    @abstractmethod
    def splits(self) -> dict[str, list[dict]]:
        """Processed QA records keyed by split name (train / val / test)."""

    @abstractmethod
    def corpus(self, states: list[str] | None = None) -> Iterator[pd.DataFrame]:
        """Legal documents for retrieval, streamed in batches."""

    @abstractmethod
    def gold_ids(self, record: dict) -> list[int]:
        """Ids of the corpus documents that support the record's answer."""
