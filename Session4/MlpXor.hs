module MlpXor where

import Control.Monad (forM_, when)
import Torch
import Torch.Layer.MLP (MLPHypParams(..), ActName(..), mlpLayer, MLPParams)
import Torch.Control (mapAccumM)
import ML.Exp.Chart (drawLearningCurve) --nlp-tools

numIters :: Int
numIters = 500

learningRate :: Tensor
learningRate = 1e-1

trainingData :: [([Float], Float)]
trainingData = [ ([0, 0], 0)
               , ([0, 1], 1)
               , ([1, 0], 1)
               , ([1, 1], 0)
               ]

trainStep ::
    Int ->
    MLPParams ->
    IO (MLPParams, Float)
trainStep i model = do
  -- Forward pass and Loss Calculation
    let loss = sum $ map (\(inputs, target) ->
                let y  = asTensor target
                    y' = mlpLayer model (asTensor inputs)
                in mseLoss y y'
            ) trainingData
        lossVal = asValue loss :: Float
    when (i `mod` 50 == 0) $
        putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show lossVal

    -- Backward pass and Weight Update
    (newModel, _) <- runStep model GD loss learningRate
    return (newModel, lossVal)

main :: IO ()
main = do
    let hypParams = MLPHypParams (Device CPU 0) 2 [(2, Tanh), (1, Tanh)]

    putStrLn "Initializing model..."
    initModel <- sample hypParams

    (trainedModel, losses) <- mapAccumM [1..numIters] initModel $ trainStep

    putStrLn "\nFinal Model Predictions:"
    forM_ trainingData $ \(input, _) -> do
        let prediction = asValue (squeezeAll $ mlpLayer trainedModel (asTensor input)) :: Float
        putStrLn $ show input ++ " => " ++ show prediction

    drawLearningCurve "Session4/img/graph-xor.png" "Learning Curve" [("",reverse losses)]
