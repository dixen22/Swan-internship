import Torch.Tensor (Tensor, asTensor)
import Torch.Functional (matmul, add, transpose2D)
import Torch.Control (mapAccumM)


ys :: [Tensor]
ys = map asTensor ([130, 195, 218, 166, 163, 155, 204, 270, 205, 127, 260, 249, 251, 158, 167] :: [Float])

xs :: [Tensor]
xs = map asTensor ([148, 186, 279, 179, 216, 127, 152, 196, 126, 78, 211, 259, 255, 115, 173] :: [Float])

linear ::
    (Tensor, Tensor) -> -- ^ parameters ([a, b]: 1 × 2, c: scalar)
    Tensor ->           -- ^ data x: 1 × 10
    Tensor              -- ^ z: 1 × 10
linear (slope, intercept) input = slope * input + intercept

displayStep :: Tensor -> Tensor -> IO ()
displayStep groundTruth estimatedY = do
    putStrLn $ "correct answer: " ++ show groundTruth
    putStrLn $ "estimated: " ++ show estimatedY
    putStrLn "******"
    return ()

linearStep :: (Tensor, Tensor) -> (Tensor, Tensor) -> IO ((Tensor, Tensor), Tensor)
linearStep (x, y) slIn = do
    let y' = linear slIn x
    displayStep y y'
    return (slIn, y')

main :: IO ()
main = do
    let a = asTensor (0.555 :: Float)
    let b = asTensor (94.585026 :: Float)

    let dataSet = zip xs ys

    (finalParams, ys') <- mapAccumM dataSet (a, b) linearStep
    print finalParams
    print ys'
    return ()