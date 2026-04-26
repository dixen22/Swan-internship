module MultipleRegression where

import Torch.Tensor (Tensor, asTensor, asValue, size)
import Torch.Functional (sumAll)
import Torch.Control (mapAccumM)
import ML.Exp.Chart (drawLearningCurve)


yList :: [Float]
yList = [123, 290, 230, 261, 140, 173, 133, 179, 210, 181]

x1List :: [Float]
x1List = [93, 230, 250, 260, 119, 183, 151, 192, 263, 185]

x2List :: [Float]
x2List = [150, 311, 182, 245, 152, 162, 99, 184, 115, 105]

linear ::
    (Tensor, Tensor, Tensor) -> -- ^ parameters ([a, b]: 1 × 2, c: scalar)
    Tensor ->
    Tensor ->    -- ^ data x: 1 × 10
    Tensor              -- ^ z: 1 × 10
linear (a1, a2, b) x1 x2 = b + (a1 * x1) + (a2 * x2)

cost ::
    Tensor -> -- ^ grand truth: 1 × 10
    Tensor ->
    Tensor    -- ^ loss: scalar
cost errors dataSize = (sumAll (errors * errors)) / (2*dataSize)

calculateNewA ::
    Tensor -> -- ^ a
    Tensor -> -- ^ lr
    Tensor -> -- ^ errors
    Tensor -> -- ^ size of x
    Tensor -> -- ^ x
    Tensor    -- ^ new a
calculateNewA a lr errors dataSize x =
    let dA = (sumAll (x * errors)) / dataSize
    in a - (lr * dA)

calculateNewB ::
    Tensor -> -- ^ b
    Tensor -> -- ^ lr
    Tensor -> -- ^ errors
    Tensor -> -- ^ size
    Tensor    -- ^ new b
calculateNewB b lr errors dataSize =
    let dB = (sumAll errors) / dataSize
    in b - (lr * dB)

trainStep :: Tensor -> Tensor -> Tensor -> Int -> ((Tensor, Tensor), (Tensor, Tensor, Tensor)) -> IO (((Tensor, Tensor), (Tensor, Tensor, Tensor)), Tensor)
trainStep x1 x2 y epoch ((lrA, lrB), (a1, a2, b)) = do
    let y' = linear (a1, a2, b) x1 x2
        dataSize = asTensor (size 0 x1)
        errors = y' - y

        newA1 = calculateNewA a1 lrA errors dataSize x1
        newA2 = calculateNewA a2 lrA errors dataSize x2
        newB = calculateNewB b lrB errors dataSize

        trainLoss = cost errors dataSize

    putStrLn $ "Epoch " ++ show epoch ++ " | Train Loss : " ++ show trainLoss
    putStrLn "******"

    return (((lrA, lrB), (newA1, newA2, newB)), trainLoss)

evalStep :: ((Float, Float), Float) -> (Tensor, Tensor, Tensor) -> IO ((Tensor, Tensor, Tensor), Float)
evalStep ((evalX1, evalX2), evalY) params = do
    let evalY' = asValue $ linear params (asTensor evalX1) (asTensor evalX2) :: Float

    putStrLn $ "correct answer: " ++ show evalY
    putStrLn $ "estimated: " ++ show evalY'
    putStrLn "******"

    return (params, evalY')

main :: IO ()
main = do
    let x1Tens = asTensor x1List
        x2Tens = asTensor x2List
        yTens = asTensor yList

    putStrLn "Train"
    putStrLn "------"
    let epochs = [1..10]
        lr = (asTensor (0.00002 :: Float), asTensor (0.015 :: Float))
        a1 = asTensor (0.0 :: Float)
        a2 = asTensor (0.0 :: Float)
        b = asTensor (0.0 :: Float)
        trainStep' = trainStep x1Tens x2Tens yTens
    ((_, (finalA1, finalA2, finalB)), lossesR) <- mapAccumM epochs (lr, (a1, a2, b)) trainStep'

    let losses = map asValue (reverse lossesR) :: [Float]

    putStrLn "Test"
    putStrLn "------"
    let xsList = zip x1List x2List
    _ <- mapAccumM (zip xsList yList) (finalA1, finalA2, finalB) evalStep

    putStrLn "Result"
    putStrLn "------"
    putStrLn $ "Final a1 : " ++ show finalA1
    putStrLn $ "Final a2 : " ++ show finalA2
    putStrLn $ "Final b : " ++ show finalB

    drawLearningCurve "Session3/img/MRLearningCurves.png" "Learning curves of MultipleRegression.hs" [("", losses)]
    putStrLn $ "Save learning curves as MRLearningCurves.png in img"

    return ()
