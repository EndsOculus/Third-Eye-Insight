from dataclasses import dataclass

import numpy as np


@dataclass
class AnalysisParams:
    L: int = 1000
    TAU: int = 150
    WINDOW_SIZE: int = 300
    BATCH_SIZE: int = 32
    N_THREADS: int = 4
    TOP_N: int = 20


@dataclass
class MetricArtifacts:
    behavior_norm: np.ndarray
    semantic_matrix_mapped: np.ndarray
    third_matrix: np.ndarray
    interaction_weights: np.ndarray
    third_metric_name: str
    behavior_title: str
    semantic_title: str
    third_title: str
