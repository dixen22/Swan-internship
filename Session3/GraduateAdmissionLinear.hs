{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE DeriveGeneric     #-}
{-# LANGUAGE LambdaCase        #-}
{-# LANGUAGE OverloadedStrings #-}

module GraduateAdmissionLinear (main) where

import GHC.Generics
import System.IO
import System.Exit (exitFailure)
import Data.Either (rights)
import Data.ByteString (ByteString, hGetSome, empty)
import Data.Csv.Incremental
import Data.Csv (FromRecord)
import Data.List.Split (splitPlaces)
import Torch.Tensor (Tensor, asTensor, asValue, size)
import Torch.Functional (sumAll, sqrt)
import Torch.Control (mapAccumM)
import ML.Exp.Chart (drawLearningCurve)

numEpochs :: Int
numEpochs = 1070

lr :: (Tensor, Tensor)
lr = (asTensor (0.015 :: Float), asTensor (0.89 :: Float))

chosenX :: String
chosenX = "cgpa" -- Chose between "greScore", "toeflScore" and "cgpa"

data Record = Record
    { serialNo         :: !Float
    , greScore         :: !Float
    , toeflScore       :: !Float
    , universityRating :: !Float
    , sop              :: !Float
    , lor              :: !Float
    , cgpa             :: !Float
    , research         :: !Float
    , chanceOfAdmit    :: !Float
    } deriving (Show, Eq, Generic)

instance FromRecord Record

feed :: (ByteString -> Parser Record) -> Handle -> IO (Parser Record)
feed k csvFile = do
    hIsEOF csvFile >>= \case
        True  -> return $ k empty
        False -> k <$> hGetSome csvFile 4096

loadFromCSV :: String -> String -> IO ([Float], [Float])
loadFromCSV path xsField = do
    withFile path ReadMode $ \ csvFile -> do
        let loop !_ (Fail _ errMsg) = do
                putStrLn $ "Erreur de parsing: " ++ errMsg
                exitFailure

            loop acc (Many rs k) =
                loop (acc <> rs) =<< feed k csvFile

            loop acc (Done rs) = do
                let allRecords = rights (acc <> rs)

                let ys = map (\r -> chanceOfAdmit r) allRecords
                    xs_gre = map (\r -> greScore r) allRecords
                    xs_toefl = map (\r -> toeflScore r) allRecords
                    xs_cgpa = map (\r -> cgpa r) allRecords

                case xsField of
                    "greScore"   -> return (xs_gre, ys)
                    "toeflScore" -> return (xs_toefl, ys)
                    "cgpa"       -> return (xs_cgpa, ys)
                    _            -> fail $ "Unknown field: " ++ xsField

        loop [] (decode HasHeader)

trainTestEvalSplit :: (Float, Float, Float) -> [e] -> ([e], [e], [e])
trainTestEvalSplit (trainPercentage, testPercentage, evalPercentage) xs =
    let xLength = fromIntegral (length xs)
        percentageToSize percentage = floor (xLength * percentage)
        trainSize = percentageToSize trainPercentage
        testSize  = percentageToSize testPercentage
        evalSize  = percentageToSize evalPercentage
        resultList = splitPlaces [trainSize, testSize, evalSize] xs
    in ((resultList !! 0), (resultList !! 1), (resultList !! 2))

getStats :: Tensor -> (Tensor, Tensor)
getStats t =
    let n = asTensor (size 0 t)
        mu = (sumAll t) / n
        diff = t - mu
        variance = (sumAll (diff * diff)) / n
        sigma = Torch.Functional.sqrt variance
    in (mu, sigma)

normalize :: Tensor -> (Tensor, Tensor) -> Tensor
normalize t (mu, sigma) = (t - mu) / sigma

linear ::
    (Tensor, Tensor) -> -- ^ parameters ([a, b]: 1 × 2, c: scalar)
    Tensor ->           -- ^ data x: 1 × 10
    Tensor              -- ^ z: 1 × 10
linear (slope, intercept) input = (slope * input) + intercept

cost ::
    Tensor -> -- ^ errors: 1 × 10
    Tensor -> -- ^ dataSize: scalar
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


trainStep :: Tensor -> Tensor -> Tensor -> Tensor -> Int -> (Tensor, Tensor) -> IO ((Tensor, Tensor), (Tensor, Tensor))
trainStep trainX trainY valX valY epoch (a, b) = do
    let trainY' = linear (a, b) trainX
        dataSize = asTensor (size 0 trainX)
        errors = trainY' - trainY

        newA = calculateNewA a (fst lr) errors dataSize trainX
        newB = calculateNewB b (snd lr) errors dataSize

        trainLoss = cost errors dataSize
        valLoss = validStep (newA, newB) valX valY

    putStrLn $ "Epoch " ++ show epoch ++ " | Train Loss : " ++ show trainLoss ++ " | Val Loss : " ++ show valLoss
    putStrLn "******"

    return ((newA, newB), (trainLoss, valLoss))

validStep :: (Tensor, Tensor) -> Tensor -> Tensor -> Tensor
validStep (a, b) valX valY =
    let valY' = linear (a, b) valX
        dataSize = asTensor (size 0 valX)
        errors = valY' - valY
    in cost errors dataSize

evalStep :: (Float, Float) -> (Tensor, Tensor) -> IO ((Tensor, Tensor), Float)
evalStep (evalX, evalY) params = do
    let evalY' = asValue $ linear params (asTensor evalX) :: Float

    putStrLn $ "correct answer: " ++ show evalY
    putStrLn $ "estimated: " ++ show evalY'
    putStrLn "******"

    return (params, evalY')

main :: IO ()
main = do
    (xList, yList) <- loadFromCSV "Session3/data/Admission_Predict.csv" chosenX

    let (trainXList, validXList, evalXList) = trainTestEvalSplit (0.8, 0.1, 0.1) xList
        (trainYList, validYList, evalYList) = trainTestEvalSplit (0.8, 0.1, 0.1) yList
        trainX = asTensor (trainXList :: [Float])
        trainY = asTensor (trainYList :: [Float])
        validX = asTensor (validXList :: [Float])
        validY = asTensor (validYList :: [Float])

    putStrLn "Train"
    putStrLn "------"
    let weights = (asTensor (0.0 :: Float), asTensor (0.0 :: Float))
    (finalWeigths, lossesR) <- mapAccumM [1..numEpochs] weights $ trainStep trainX trainY validX validY

    let losses = reverse lossesR
        trainLoss = [asValue x :: Float | (x, _) <- losses]
        validLoss = [asValue x :: Float | (_, x) <- losses]

    putStrLn "Test"
    putStrLn "------"
    _ <- mapAccumM (zip evalXList evalYList) finalWeigths evalStep

    putStrLn "Result"
    putStrLn "------"
    putStrLn $ "Final a : " ++ show (fst finalWeigths)
    putStrLn $ "Final b : " ++ show (snd finalWeigths)

    let imgName = "GALLearningCurves-" ++ chosenX ++ ".png"
        title = "Learning curves of GraduateAdmissionLinear.hs with " ++ chosenX ++ " in x"

    drawLearningCurve ("Session3/img/" ++ imgName) title [("Train", trainLoss), ("Valid", validLoss)]
    putStrLn $ "Save learning curves as " ++ imgName ++ " in img"

    return ()
