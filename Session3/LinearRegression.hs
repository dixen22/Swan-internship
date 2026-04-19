import Torch.Tensor (Tensor, asTensor, asValue)
import Torch.Functional (matmul, add, transpose2D)


ys :: [Float]
ys = [130, 195, 218, 166, 163, 155, 204, 270, 205, 127, 260, 249, 251, 158, 167]
xs :: [Float]
xs = [148, 186, 279, 179, 216, 127, 152, 196, 126, 78, 211, 259, 255, 115, 173]

linear ::
    (Tensor, Tensor) -> -- ^ parameters ([a, b]: 1 × 2, c: scalar)
    Tensor ->           -- ^ data x: 1 × 10
    Tensor              -- ^ z: 1 × 10
linear (slope, intercept) input = slope * input + intercept

displayResults :: Show a => [(a, a)] -> IO ()
displayResults [] = return ()
displayResults ((groundTruth, estimatedY):l) = do
    putStrLn ("correct answer: " ++ show groundTruth)
    putStrLn ("estimated: " ++ show estimatedY)
    putStrLn "******"
    displayResults l

main :: IO ()
main = do
    let a = asTensor (0.555 :: Float)
    let b = asTensor (94.585026 :: Float)

    let x = asTensor xs
    let y' = linear (a, b) x

    displayResults (zip ys (asValue y' :: [Float]))
    return ()