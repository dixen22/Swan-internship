module LinearRegression where

import Torch.Tensor (Tensor, asTensor, asValue, size)
import Torch.Functional (sumAll)
import Torch.Control (mapAccumM)
import ML.Exp.Chart (drawLearningCurve)


ys :: [Float]
ys = [130, 195, 218, 166, 163, 155, 204, 270, 205, 127, 260, 249, 251, 158, 167]

xs :: [Float]
xs = [148, 186, 279, 179, 216, 127, 152, 196, 126, 78, 211, 259, 255, 115, 173]

linear ::
    (Tensor, Tensor) -> -- ^ parameters ([a, b]: 1 × 2, c: scalar)
    Tensor ->           -- ^ data x: 1 × 10
    Tensor              -- ^ z: 1 × 10
linear (slope, intercept) input = (slope * input) + intercept

cost ::
    Tensor -> -- ^ grand truth: 1 × 10
    Tensor ->
    Tensor    -- ^ loss: scalar
cost errors dataSize = (sumAll (errors * errors)) / (2*dataSize)

calculateNewA ::
    Tensor -> -- ^ a
    Tensor -> -- ^ errors
    Tensor -> -- ^ size of x
    Tensor -> -- ^ x
    Tensor
calculateNewA a errors dataSize x =
    let dA = (sumAll (x * errors)) / dataSize
    in a - (0.0000255 * dA)

calculateNewB ::
    Tensor -> -- ^ b
    Tensor -> -- ^ errors
    Tensor -> -- ^ size
    Tensor
calculateNewB b errors dataSize =
    let dB = (sumAll errors) / dataSize
    in b - (0.82 * dB)

displayStep ::
    Float ->
    Float ->
    IO ()
displayStep y yPred = do
    putStrLn $ "correct answer: " ++ show y
    putStrLn $ "estimated: " ++ show yPred
    putStrLn "******"
    return ()

linearStep ::
    (Float, Float) ->
    (Tensor, Tensor) ->
    IO ((Tensor, Tensor), Float)
linearStep (x, y) params = do
    let yPred = asValue $ linear params (asTensor x) :: Float
    displayStep y yPred
    return (params, yPred)

linearStep' ::
    (Float, Float) ->
    (Tensor, Tensor) ->
    IO ((Tensor, Tensor), Float)
linearStep' (x, _) params = do
    let yPred = asValue $ linear params (asTensor x) :: Float
    return (params, yPred)

trainStep :: Int -> (Tensor, Tensor) -> IO ((Tensor, Tensor), Tensor)
trainStep epoch (a, b) = do
    let dataSet = zip xs ys
        xsTens = asTensor(xs)
        ysTens = asTensor(ys)

    (_, yPredsR) <- mapAccumM dataSet (a, b) linearStep'

    let yPreds = asTensor (reverse yPredsR)
        dataSize = asTensor (size 0 xsTens)
        errors = yPreds - ysTens

    let loss = cost errors dataSize
        newA = calculateNewA a errors dataSize xsTens
        newB = calculateNewB b errors dataSize

    putStrLn $ "Epoch " ++ show epoch ++ " | Loss : " ++ show loss
    putStrLn "******"

    return ((newA, newB), loss)

main :: IO ()
main = do
    let a = asTensor (0 :: Float)
        b = asTensor (0 :: Float)

    putStrLn "Train"
    putStrLn "------"
    let epochs = [1..15]
    ((finalA, finalB), lossesR) <- mapAccumM epochs (a, b) trainStep

    let losses = map asValue (reverse lossesR) :: [Float]

    putStrLn "Test"
    putStrLn "------"
    _ <- mapAccumM (zip xs ys) (finalA, finalB) linearStep

    putStrLn "Result"
    putStrLn "------"
    putStrLn $ "Final a : " ++ show finalA
    putStrLn $ "Final b : " ++ show finalB


    drawLearningCurve "Session3/img/LRLearningCurve.png" "Learning curve of LinearRegression.hs" [("", losses)]
    putStrLn "Save learning curve as LRLearningCurve.png in img"

    return ()
