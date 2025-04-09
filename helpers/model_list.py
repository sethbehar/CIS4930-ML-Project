from models.decisionTree import decisionTree
from models.logisticRegression import logisticRegression
from models.kNearestNeighbors import kNearestNeighbors


# model dictionary for pipelines
model_map = {
    "Decision Tree": decisionTree,
    "Logistic Regression": logisticRegression,
    "kNearestNeighbors": kNearestNeighbors
}