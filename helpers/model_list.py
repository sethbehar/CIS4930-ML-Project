from models.decisionTree import decisionTree
from models.logisticRegression import logisticRegression
from models.kNearestNeighbors import kNearestNeighbors
from models.svc import svc
from models.catBoost import catBoost
from models.neuralNetwork import neuralNetwork
from models.adaboost import adaboost
from models.bagging import bagging
from models.gradientBoosting import gradientBoosting
from models.randomForest import randomForest
from models.stacking import stacking

# model dictionary for pipelines
model_map = {
    "Neural Network": neuralNetwork,
    "Cat Boost": catBoost,
    "Decision Tree": decisionTree,
    "Logistic Regression": logisticRegression,
    "K Nearest Neighbors": kNearestNeighbors,
    "AdaBoost": adaboost,
    "Bagging": bagging,
    "Gradient Boosting": gradientBoosting,
    "Random Forest": randomForest,
    "Stacking": stacking
}