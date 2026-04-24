{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE DeriveGeneric     #-}
{-# LANGUAGE LambdaCase        #-}
{-# LANGUAGE OverloadedStrings #-}

module GraduateAdmissionLinear where

import GHC.Generics
import System.IO
import System.Exit (exitFailure)
import Data.Either (rights)
import Data.ByteString (ByteString, hGetSome, empty)
import Data.Csv.Incremental
import Data.Csv (FromRecord)
import Torch.Tensor (Tensor, asTensor, asValue, size)
import Torch.Functional (sumAll, sqrt)
import Torch.Control (mapAccumM)
import ML.Exp.Chart (drawLearningCurve)


data Record = Record
    { serialNo      :: !Float
    , chanceOfAdmit :: !Float
    , greScore      :: !Float
    , toeflScore    :: !Float
    , cgpa          :: !Float
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


trainStep :: Tensor -> Tensor -> Tensor -> Tensor -> Int -> ((Tensor, Tensor), (Tensor, Tensor)) -> IO (((Tensor, Tensor), (Tensor, Tensor)), (Tensor, Tensor))
trainStep trainX trainY valX valY epoch ((lrA, lrB), (a, b)) = do
    let trainY' = linear (a, b) trainX
        dataSize = asTensor (size 0 trainX)
        errors = trainY' - trainY

        newA = calculateNewA a lrA errors dataSize trainX
        newB = calculateNewB b lrB errors dataSize

        trainLoss = cost errors dataSize
        valLoss = validStep (newA, newB) valX valY

    putStrLn $ "Epoch " ++ show epoch ++ " | Train Loss : " ++ show trainLoss ++ " | Val Loss : " ++ show valLoss
    putStrLn "******"

    return (((lrA, lrB), (newA, newB)), (trainLoss, valLoss))

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
    let chosenX = "cgpa" -- Chose between "greScore", "toeflScore" and "cgpa"*
        normalizeX = False
    (trainXList, trainYList) <- loadFromCSV "Session3/data/train.csv" chosenX
    (validXList, validYList) <- loadFromCSV "Session3/data/valid.csv" chosenX
    (evalXList', evalYList) <- loadFromCSV "Session3/data/eval.csv" chosenX

    let rawTrainX = asTensor (trainXList :: [Float])
        statsTrain = getStats rawTrainX
        trainX = if normalizeX
                    then normalize rawTrainX statsTrain
                    else rawTrainX
        trainY = asTensor (trainYList :: [Float])
        validX = if normalizeX
                    then normalize (asTensor (validXList :: [Float])) statsTrain
                    else asTensor (validXList :: [Float])
        validY = asTensor (validYList :: [Float])
        evalXList = if normalizeX
                        then asValue (normalize (asTensor (evalXList' :: [Float])) statsTrain) :: [Float]
                        else evalXList'

    putStrLn "Train"
    putStrLn "------"
    let epochs = [1..1070]
        lr = (asTensor (0.015 :: Float), asTensor (0.89 :: Float))
        a = asTensor (0.0 :: Float)
        b = asTensor (0.0 :: Float)
        trainStep' = trainStep trainX trainY validX validY
    ((_, (finalA, finalB)), lossesR) <- mapAccumM epochs (lr, (a, b)) trainStep'

    let losses = reverse lossesR
        trainLoss = [asValue x :: Float | (x, _) <- losses]
        validLoss = [asValue x :: Float | (_, x) <- losses]

    putStrLn "Test"
    putStrLn "------"
    _ <- mapAccumM (zip evalXList evalYList) (finalA, finalB) evalStep

    putStrLn "Result"
    putStrLn "------"
    putStrLn $ "Final a : " ++ show finalA
    putStrLn $ "Final b : " ++ show finalB

    let imgName = "GALLearningCurves-" ++ chosenX ++ ".png"
        title = "Learning curves of GraduateAdmissionLinear.hs with " ++ chosenX ++ " in x"

    drawLearningCurve ("Session3/img/" ++ imgName) title [("Train", trainLoss), ("Valid", validLoss)]
    putStrLn $ "Save learning curves as " ++ imgName ++ " in img"

    return ()
