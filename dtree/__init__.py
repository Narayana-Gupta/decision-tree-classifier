"""dtree — from-scratch ID3 and C4.5 decision trees with preprocessing, evaluation and plotting.

Contributors: Karthik Naidu Allu (2023002399), Boddu Narayana Gupta (2023002459)
"""
from .base import BaseDecisionTree, Node, entropy
from .c45 import C45Classifier
from .id3 import ID3Classifier

__all__ = ["BaseDecisionTree", "Node", "entropy", "ID3Classifier", "C45Classifier"]
__version__ = "1.0.0"

CONTRIBUTORS = [
    ("Karthik Naidu Allu", "2023002399"),
    ("Boddu Narayana Gupta", "2023002459"),
]
