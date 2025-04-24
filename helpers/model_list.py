from models.decisionTree import decisionTree
from models.logisticRegression import logisticRegression
from models.kNearestNeighbors import kNearestNeighbors
from models.neuralNetwork import neuralNetwork
from models.catboost import catBoost
from models.svc import svc
from models.adaboost import adaboost
from models.bagging import bagging
from models.gradientBoosting import gradientBoosting
from models.randomForest import randomForest
from models.stacking import stacking
# model dictionary for pipelines
model_map = {
    "Decision Tree": decisionTree,
    "Logistic Regression": logisticRegression,
    "K Nearest Neighbors": kNearestNeighbors,
    "Neural Network": neuralNetwork,
    "Cat Boost": catBoost,
    "AdaBoost": adaboost,
    "Bagging": bagging,
    "Gradient Boosting": gradientBoosting,
    "Random Forest": randomForest,
    "Stacking": stacking,
    "SVC": svc
}