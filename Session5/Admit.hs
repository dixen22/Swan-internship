{-# LANGUAGE DeriveGeneric #-}

{-|
Module      : Admit
Description : Entry point for the University Admission prediction model.

This script wires together the CSV loader and MLP components.
It defines the specific schema for the Admission dataset and the hyperparameters
used to configure the neural network.
-}
module Admit (main) where

import GHC.Generics
import Data.Csv (FromRecord)
import LoadData (loadFromCSV)
import Torch
import Torch.Layer.MLP (MLPHypParams(..), ActName(..))
import Torch.Control (mapAccumM)
import Mlp (trainStep, evalStep)
import Evaluation (evaluate)
import ML.Exp.Chart (drawLearningCurve)

-- | Represents a single row in the admission dataset.
data AdmitRecord = AdmitRecord
    { serialNo      :: !Float
    , chanceOfAdmit :: !Float
    , greScore      :: !Float
    , toeflScore    :: !Float
    , cgpa          :: !Float
    } deriving (Show, Eq, Generic)

instance FromRecord AdmitRecord

-- | Extracts model features and the target label from an admission record.
-- Feature selection: We currently use GRE, TOEFL, and CGPA as predictors.
extractAdmitData :: AdmitRecord -> ([Float], [Float])
extractAdmitData r = ([greScore r, toeflScore r, cgpa r], [chanceOfAdmit r])

-- | Total number of training passes over the dataset.
numEpoch :: Int
numEpoch = 250

-- | Static learning rate. Kept relatively high (0.01) for fast initial convergence.
learningRate :: Tensor
learningRate = 9e-1

-- | Defines the architecture of the Multilayer Perceptron.
-- We use an input layer (size 3), one hidden layer (size 8), and an output layer (size 1).
-- Sigmoid activation is used because the target 'chance of admit' is bounded between 0 and 1.
hypParams :: MLPHypParams
hypParams = MLPHypParams (Device CPU 0) 3 [(4, Sigmoid), (1, Sigmoid)]

-- | Calculates the error between predictions and target labels.
-- Binary Cross Entropy is appropriate here since outputs are probabilistic (0 to 1).
lossFunction :: Tensor -> Tensor -> Tensor
lossFunction y y' = mseLoss (squeezeAll y) (squeezeAll y')

threshold :: Float
threshold = 0.7

-- | Orchestrates data loading, model training, evaluation, and chart plotting.
main :: IO ()
main = do
    -- Inject the domain-specific extractor to load standard tensors
    trainData <- loadFromCSV "Session3/data/train.csv" extractAdmitData
    validData <- loadFromCSV "Session3/data/valid.csv" extractAdmitData
    evalData  <- loadFromCSV "Session3/data/eval.csv" extractAdmitData

    putStrLn "Train"
    putStrLn "------"
    -- Initialize random weights based on the defined architecture
    initModel <- sample hypParams

    -- mapAccumM threads the model state through numEpoch iterations automatically
    (trainedModel, lossesR) <- mapAccumM [1..numEpoch] initModel $ trainStep lossFunction learningRate trainData validData

    -- Reverse loss lists because mapAccumM prepends outputs for efficiency
    let losses    = reverse lossesR
        trainLoss = [x :: Float | (x, _) <- losses]
        validLoss = [x :: Float | (_, x) <- losses]

    putStrLn "\nTest"
    putStrLn "------"
    (_, evalPred) <- evalStep evalData trainedModel

    putStrLn "\nResult"
    putStrLn "------"
    let evalActual       = asValue (squeezeAll (snd evalData)) :: [Float]
        toClass x        = if x >= Admit.threshold then 1 else 0
        actualClasses    = map toClass evalActual
        predictedClasses = map toClass evalPred

    evaluate [0, 1] actualClasses predictedClasses
    drawLearningCurve "Session5/img/admit-learning-curve.png" "Learning Curve" [("train", trainLoss), ("valid", validLoss)]
    putStrLn "\nLearning curve saved"
