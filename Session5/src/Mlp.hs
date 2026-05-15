{-|
Module      : Mlp
Description : Core training and evaluation logic for the Multilayer Perceptron.

Provides domain-agnostic functions for forward passes, backward passes, and evaluation.
Loss functions and learning rates are injected as dependencies to maximize code reusability.
-}
module Mlp (trainStep, evalStep) where

import Torch
import Torch.Layer.MLP (MLPParams(..), mlpLayer)
import Control.Monad (when)
import Text.Printf (printf)

-- | Executes a single training iteration, computing both training and validation losses.
-- Weights are updated using Standard Gradient Descent (GD).
trainStep :: (Tensor -> Tensor -> Tensor) -- ^ Loss function (e.g., Binary Cross Entropy)
          -> Tensor                       -- ^ Learning rate defining step size
          -> (Tensor, Tensor)             -- ^ Training set containing (Features, Targets)
          -> (Tensor, Tensor)             -- ^ Validation set containing (Features, Targets)
          -> Int                          -- ^ Current epoch number (used for logging)
          -> MLPParams                    -- ^ Current state of the model parameters
          -> IO (MLPParams, (Float, Float))
trainStep lossFn lr (xTrain, yTrain) (xVal, yVal) epoch model = do

    -- Compute predictions and measure error against actual targets
    let y'Train = mlpLayer model xTrain
        trainLoss = lossFn yTrain y'Train
        trainLossValue = asValue trainLoss :: Float

    -- Backpropagate the error to adjust model weights
    (newModel, _) <- runStep model GD trainLoss lr

    -- Validate the updated model on unseen data to monitor for overfitting
    let y'Val = mlpLayer newModel xVal
        valLoss = lossFn yVal y'Val
        valLossValue = asValue valLoss :: Float

    -- Log progress periodically rather than every epoch to reduce standard output noise
    when (epoch `mod` 50 == 0) $
        printf "  Epoch %4d | Train Loss: %12.6e | Val Loss: %12.6e\n" epoch trainLossValue valLossValue

    return (newModel, (trainLossValue, valLossValue))

-- | Evaluates the fully trained model against a test dataset and logs a sample of predictions.
evalStep :: (Tensor, Tensor) -- ^ Evaluation set (Features, Targets)
         -> MLPParams        -- ^ Fully trained model
         -> IO (MLPParams, [Float])
evalStep (xEval, yEval) trainedModel = do
    -- The squeeze operation flattens output tensors into standard lists for easier comparison
    let predictions = asValue (squeezeAll $ mlpLayer trainedModel xEval) :: [Float]
        targets     = asValue (squeezeAll yEval) :: [Float]

    -- Print the first 10 predictions vs actuals for a quick visual sanity check
    mapM_ (\(p, t) -> printf "  Predicted: %6.4f | actual: %4.2f\n" p t) (Prelude.take 10 $ zip predictions targets)

    return (trainedModel, predictions)
