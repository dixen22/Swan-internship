module PerceptronAndGate where

import Torch.Tensor (Tensor, asTensor, asValue)
import Torch.TensorFactories (randIO')
import Torch.Functional (dot)
import Torch.Control (mapAccumM)

trainingData :: [([Int],Int)]
trainingData = [([1,1],1),([1,0],0),([0,1],0),([0,0],0)]

learningRate :: Float
learningRate = 0.1
maxIter :: Int
maxIter = 1000

step :: Tensor -> Tensor
step x = if (asValue x :: Float) > 0
            then 1
            else 0

perceptron ::
    Tensor -> -- x
    Tensor -> -- weights
    Tensor -> -- bias
    Tensor    -- output
perceptron x w b = step (x `dot` w + b)

caluculateError ::
    Tensor ->
    Tensor ->
    Tensor
caluculateError y y' = y - y'

trainStep ::
    ([Int], Int) ->
    (Tensor, Tensor) ->
    IO ((Tensor, Tensor), Float)
trainStep (xList, yVal) (cw, cb) = do
      let x = asTensor (map fromIntegral xList :: [Float])
          y = asTensor (fromIntegral yVal :: Float)

          y' = perceptron x cw cb

          err = caluculateError y y'
          errVal = asValue err :: Float

          cw' = cw + (asTensor (learningRate * errVal) * x)
          cb' = cb + asTensor (learningRate * errVal)

      return ((cw', cb'), errVal)

trainLoop ::
    Int ->
    Tensor ->
    Tensor ->
    IO ()
trainLoop epoch currentW currentB
      | epoch >= maxIter = putStrLn "Max iterations reached."
      | otherwise = do
          ((nextW, nextB), errors) <- mapAccumM trainingData (currentW, currentB) trainStep

          let totalError = sum [abs e | e <- errors]
          putStrLn $ "Epoch " ++ show epoch ++ " | Total Error: " ++ show totalError

          if totalError == 0.0
              then putStrLn $ "End at epoch " ++ show epoch ++ " with weights " ++ show (asValue nextW :: [Float]) ++ " and bias " ++ show (asValue nextB :: Float)
              else trainLoop (epoch + 1) nextW nextB

main :: IO ()
main = do
    w <- randIO' [2]
    b <- randIO' []

    -- train!
    trainLoop 1 w b
    return ()
