    from models.decisionTree import decisionTree
from models.logisticRegression import logisticRegression
from models.kNearestNeighbors import kNearestNeighbors
from models.svc import svc
from models.neuralNetwork import neuralNetwork
# model dictionary for pipelines
model_map = {
    "Decision Tree": decisionTree,
    "Logistic Regression": logisticRegression,
    "K Nearest Neighbors": kNearestNeighbors,
    "SVC": svc,
    "Neural Network": neuralNetwork
}