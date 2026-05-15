{-# LANGUAGE DeriveGeneric #-}

{-|
Module      : Titanic
Description : Entry point for the Titanic prediction model.

This script wires together the CSV loader and MLP components.
It defines the specific schema for the Titanic dataset and the hyperparameters
used to configure the neural network.
-}
module Titanic (main) where

import GHC.Generics
import Data.Csv (FromRecord)
import LoadData (loadFromCSV)
import Torch
import Torch.Layer.MLP (MLPHypParams(..), ActName(..))
import Torch.Control (mapAccumM)
import Mlp (trainStep, evalStep)
import Evaluation (evaluate)
import ML.Exp.Chart (drawLearningCurve)

data TitanicRecord = TitanicRecord
    { passengerId :: !String
    , survived    :: !String
    , pclass      :: !String
    , name        :: !String
    , sex         :: !String
    , age         :: !String
    , sibSp       :: !String
    , parch       :: !String
    , ticket      :: !String
    , fare        :: !String
    , cabin       :: !String
    , embarked    :: !String
    } deriving (Show, Eq, Generic)

instance FromRecord TitanicRecord

parseFloat :: String -> Float -> Float
parseFloat "" defaultVal = defaultVal
parseFloat s  _          = read s

extractTitanicData :: TitanicRecord -> ([Float], [Float])
extractTitanicData r =
    let sexEncoded = if sex r == "female" then 1.0 else 0.0

        ageImputed  = parseFloat (age r) 0.0
        fareImputed = parseFloat (fare r) 0.0
        pclassImputed = parseFloat (pclass r) 0.0
        sibSpImputed = parseFloat (sibSp r) 0.0
        parchImputed = parseFloat (parch r) 0.0
        survivedImputed = parseFloat (survived r) 0.0

        features =
            [ pclassImputed
            , sexEncoded
            , ageImputed
            , sibSpImputed
            , parchImputed
            , fareImputed
            ]

        targets = [survivedImputed]
    in (features, targets)

numEpoch :: Int
numEpoch = 100

learningRate :: Tensor
learningRate = 1e-1

hypParams :: MLPHypParams
hypParams = MLPHypParams (Device CPU 0) 6 [(8, Relu), (4, Relu), (1, Sigmoid)]

lossFunction :: Tensor -> Tensor -> Tensor
lossFunction actual prediction = binaryCrossEntropyLoss' (squeezeAll actual) (squeezeAll prediction)

threshold :: Float
threshold = 0.5

main :: IO ()
main = do
    trainData <- loadFromCSV "Session5/data/train.csv" extractTitanicData
    validData <- loadFromCSV "Session5/data/valid.csv" extractTitanicData
    evalData  <- loadFromCSV "Session5/data/eval.csv" extractTitanicData

    putStrLn "Train"
    putStrLn "------"
    initModel <- sample hypParams

    (trainedModel, lossesR) <- mapAccumM [1..numEpoch] initModel $ trainStep lossFunction learningRate trainData validData

    let losses    = reverse lossesR
        trainLoss = [x :: Float | (x, _) <- losses]
        validLoss = [x :: Float | (_, x) <- losses]

    putStrLn "\nTest"
    putStrLn "------"
    (_, evalPred) <- evalStep evalData trainedModel

    putStrLn "\nResult"
    putStrLn "------"
    let evalActual       = asValue (squeezeAll (snd evalData)) :: [Float]
        toClass x        = if x >= Titanic.threshold then 1 else 0
        actualClasses    = map toClass evalActual
        predictedClasses = map toClass evalPred

    evaluate [0, 1] actualClasses predictedClasses
    drawLearningCurve "Session5/img/titanic-learning-curve.png" "Learning Curve" [("train", trainLoss), ("valid", validLoss)]
    putStrLn "\nLearning curve saved"
