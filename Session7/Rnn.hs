{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Rnn (main) where

import Codec.Binary.UTF8.String (encode)
import Data.Aeson (FromJSON(..), ToJSON(..), eitherDecode)
import qualified Data.ByteString.Lazy as B
import qualified Data.ByteString.Internal as B (c2w)
import qualified Data.ByteString.Lazy.Char8 as B8
import qualified Data.Map.Strict as M
import GHC.Generics
import Data.Int (Int64)
import Data.List (foldl')
import Data.Char (toLower, isAlphaNum)
import Data.List.Split (splitPlaces)
import Control.Monad (when)
import Text.Printf (printf)
import System.IO (hFlush, stdout)
import System.Mem (performMinorGC)

import Torch.NN (Parameter, Parameterized(..), Randomizable(..), sample)
import Torch.Serialize (loadParams, saveParams)
import Torch.TensorFactories (randnIO', zeros')
import Torch.Autograd (makeIndependent, toDependent)
import Torch.Functional (embedding')
import Torch.Tensor (Tensor, asTensor, asValue)
import Torch.Layer.Linear (LinearHypParams(..), LinearParams(..), linearLayer)
import Torch.Optim (GD(..), runStep)
import Torch.Control (makeBatch, foldLoopM)
import ML.Exp.Chart (drawLearningCurve)
import qualified Torch as T
import Evaluation (evaluate)

embeddingIsRandom :: Bool
embeddingIsRandom = False

embeddingDimSize :: Int
embeddingDimSize = 128

hiddenDimSize :: Int
hiddenDimSize = 32

batchSize :: Int
batchSize = 32

maxEpoch :: Int
maxEpoch = 50

learningRate :: Tensor
learningRate = asTensor (0.0001 :: Float)

patience :: Int
patience = 3

delta :: Float
delta = 0.05

trainReviewPath, validReviewPath, testReviewPath :: FilePath
trainReviewPath = "Session7/data/train.jsonl"
validReviewPath = "Session7/data/valid.jsonl"
testReviewPath  = "Session7/data/test.jsonl"

embeddingPath, wordLstPath :: FilePath
embeddingPath = "Session6/data/sample_embedding.params"
wordLstPath   = "Session6/data/sample_wordlst.txt"

modelOutPath, imgPath :: FilePath
modelOutPath = "Session7/data/rnn_model.params"
imgPath      = "Session7/img/learning-curve.png"

data Image = Image {
  small_image_url :: String,
  medium_image_url :: String,
  large_image_url :: String
} deriving (Show, Generic, FromJSON, ToJSON)

data AmazonReview = AmazonReview {
  rating :: Float,
  title :: String,
  text :: String,
  images :: [Image],
  asin :: String,
  parent_asin :: String,
  user_id :: String,
  timestamp :: Int,
  verified_purchase :: Bool,
  helpful_vote :: Int
} deriving (Show, Generic, FromJSON, ToJSON)

data ModelSpec = ModelSpec {
  wordNum :: Int,
  wordDim :: Int
} deriving (Show, Eq, Generic)

data RnnSpec = RnnSpec {
  inputDim  :: Int,
  hiddenDim :: Int
} deriving (Show, Eq, Generic)

data Embedding = Embedding {
    wordEmbedding :: Parameter
} deriving (Show, Generic, Parameterized)

data RNN = RNN {
  input_weight  :: Parameter,
  hidden_weight :: Parameter,
  bias          :: Parameter
} deriving (Show, Generic, Parameterized)

data Model = Model {
  emb     :: Embedding,
  rnn     :: RNN,
  decoder :: LinearParams
} deriving (Show, Generic, Parameterized)

instance Randomizable RnnSpec RNN where
  sample RnnSpec {..} = do
    w_ih_raw <- randnIO' [inputDim, hiddenDim]
    w_hh_raw <- randnIO' [hiddenDim, hiddenDim]
    let b_raw = zeros' [hiddenDim]

    w_ih <- makeIndependent $ T.mulScalar (0.1 :: Float) w_ih_raw
    w_hh <- makeIndependent $ T.mulScalar (0.1 :: Float) w_hh_raw
    b    <- makeIndependent b_raw

    return $ RNN w_ih w_hh b

instance Randomizable ModelSpec Model where
    sample ModelSpec {..} =
        Model
        <$> (Embedding <$> (makeIndependent =<< randnIO' [wordNum, wordDim]))
        <*> sample (RnnSpec wordDim hiddenDimSize)
        <*> sample (LinearHypParams (T.Device T.CPU 0) True hiddenDimSize 1)

initialize :: ModelSpec -> FilePath -> IO Model
initialize modelSpec embPath = do
  randomizedModel <- sample modelSpec
  loadedEmb <- loadParams (emb randomizedModel) embPath
  if embeddingIsRandom
    then return Model { emb = loadedEmb, rnn = rnn randomizedModel, decoder = decoder randomizedModel }
    else do
      return randomizedModel

tanhFunc :: Tensor -> Tensor
tanhFunc x = (T.exp x - T.exp (-x)) / (T.exp x + T.exp (-x))

gate :: Tensor -> Tensor -> (Tensor -> Tensor) -> Tensor -> Tensor -> Tensor -> Tensor
gate input hidden activ w_ih w_hh b = activ (T.matmul input w_ih + T.matmul hidden w_hh + b)

nextState :: RNN -> Tensor -> Tensor -> Tensor
nextState RNN {..} input hidden =
  let ih = toDependent input_weight
      hh = toDependent hidden_weight
      b  = toDependent bias
  in gate input hidden tanhFunc ih hh b

unstack :: Tensor -> [Tensor]
unstack t = [T.select 0 i t | i <- [0 .. (head (T.shape t) - 1)]]

forwardRegression :: Model -> Tensor -> [Int64] -> Tensor
forwardRegression model h0 wordIds =
  let xTrain = asTensor wordIds
      wEmb = toDependent (wordEmbedding (emb model))
      embTrain = embedding' wEmb xTrain
      wordVectors = unstack embTrain
      hLast = foldl' (\hBrut x_t -> nextState (rnn model) x_t hBrut) h0 wordVectors
  in linearLayer (decoder model) hLast

type Batch = [([Int64], Float)]

mseLoss :: Tensor -> Tensor -> Tensor
mseLoss prediction target = T.mean ((prediction - target) * (prediction - target))

calcBatchLoss :: Model -> Batch -> Tensor
calcBatchLoss model batch =
    let totalLoss = foldl' (\accLoss (wordIds, targetRating) ->
         let h0 = zeros' [hiddenDimSize]
             prediction = forwardRegression model h0 wordIds
             target = asTensor [targetRating]
         in accLoss + mseLoss prediction target ) (zeros' [1]) batch
    in totalLoss / asTensor [fromIntegral (length batch) :: Float]

trainBatch :: Int -> (Int, Batch) -> (Model, Float) -> IO (Model, Float)
trainBatch totalBatches (batchIdx, batch) (currModel, !totalLoss) = do
    let batchLoss = calcBatchLoss currModel batch
    (newModel, _) <- runStep currModel GD batchLoss learningRate

    when (batchIdx `mod` 50 == 0 || batchIdx == totalBatches) $ do
        printf "\r\ESC[K  Training %d/%d" batchIdx totalBatches
        hFlush stdout

    performMinorGC
    return (newModel, totalLoss + (asValue batchLoss :: Float))

valBatch :: Int -> (Int, Batch) -> (Model, Float) -> IO (Model, Float)
valBatch totalBatches (batchIdx, batch) (currModel, !totalLoss) = do
    let batchLoss = calcBatchLoss currModel batch

    when (batchIdx `mod` 50 == 0 || batchIdx == totalBatches) $ do
        printf "\r\ESC[K  Validation %d/%d" batchIdx totalBatches
        hFlush stdout

    performMinorGC
    return (currModel, totalLoss + (asValue batchLoss :: Float))

processEpoch :: [(Int, Batch)] -> [(Int, Batch)] -> Int -> Model -> IO (Model, (Float, Float))
processEpoch trainBatches valBatches epoch model = do
    putStrLn $ "Epoch " ++ show epoch
    let totalTrain = length trainBatches
        totalVal = length valBatches

    (newModel, totalTrainLoss) <- foldLoopM trainBatches (model, 0.0) (trainBatch totalTrain)
    putStrLn ""
    (_, totalValLoss) <- foldLoopM valBatches (newModel, 0.0) (valBatch totalVal)

    let avgTrainLoss = totalTrainLoss / fromIntegral totalTrain
        avgValLoss = totalValLoss / fromIntegral totalVal

    printf "\n  Train Loss: %12.6e | Val Loss: %12.6e\n" avgTrainLoss avgValLoss
    hFlush stdout
    return (newModel, (avgTrainLoss, avgValLoss))

trainLoop :: [(Int, Batch)] -> [(Int, Batch)] -> Int -> Int -> Float -> Model -> [(Float, Float)] -> IO (Model, [(Float, Float)])
trainLoop trainBatches valBatches epoch patienceCounter bestValLoss model losses
    | epoch > maxEpoch || patienceCounter >= patience = do
        putStrLn "\nFin de l'entraînement (Early Stopping ou Max Epoch atteint)."
        return (model, reverse losses)
    | otherwise = do
        (newModel, (trainLoss, valLoss)) <- processEpoch trainBatches valBatches epoch model
        let !newLosses = (trainLoss, valLoss) : losses
            nextBestLoss = min valLoss bestValLoss
            nextPatience = if valLoss < bestValLoss - delta then 0 else patienceCounter + 1

        trainLoop trainBatches valBatches (epoch + 1) nextPatience nextBestLoss newModel newLosses

keepSplit :: Float -> [e] -> [e]
keepSplit trainRatio xs =
    let xLength = length xs
        trainSize = round (fromIntegral xLength * trainRatio)
        endSize = xLength - trainSize
        resultList = splitPlaces [trainSize, endSize] xs
    in head resultList

preprocess :: B.ByteString -> [[B.ByteString]]
preprocess texts = map (map (B8.filter isAlphaNum) . B8.words) textLines
  where
    filteredtexts = B.filter (\w -> w `notElem` map (head . encode) [".", "!"]) texts
    textLines = B8.lines (B8.map toLower filteredtexts)

wordToIndexFactory :: [B.ByteString] -> (B.ByteString -> Int64)
wordToIndexFactory wordlst wrd =
  M.findWithDefault (fromIntegral (length wordlst)) wrd (M.fromList (zip wordlst [0..]))

decodeToAmazonReview :: B.ByteString -> Either String [AmazonReview]
decodeToAmazonReview jsonl = sequenceA $ map eitherDecode (filter (not . B.null) (B.split (B.c2w '\n') jsonl))

loadDataset :: FilePath -> IO [AmazonReview]
loadDataset path = do
  jsonl <- B.readFile path
  case decodeToAmazonReview jsonl of
    Left _   -> return []
    Right r  -> return r

cleanDataset :: (B.ByteString -> Int64) -> [AmazonReview] -> [([Int64], Float)]
cleanDataset wordToIndex = filter (\(ids, _) -> not (null ids)) . map (\r ->
    let tokens = take 50 $ concat $ preprocess (B8.pack $ text r)
        ids = map wordToIndex tokens
    in (ids, rating r))


main :: IO ()
main = do
    trainData' <- loadDataset trainReviewPath
    validData' <- loadDataset validReviewPath
    testData'  <- loadDataset testReviewPath

    wordLst <- fmap (B.split (head $ encode "\n")) (B.readFile wordLstPath)
    let wordToIndex = wordToIndexFactory wordLst
        totalWords  = length wordLst + 1
        modelSpec = ModelSpec { wordDim = embeddingDimSize, wordNum = totalWords }

    let cleanTrainData = cleanDataset wordToIndex trainData'
        cleanValidData = cleanDataset wordToIndex validData'
        cleanTestData  = cleanDataset wordToIndex testData'

    let !trainData = keepSplit 0.02 cleanTrainData
        !validData = keepSplit 0.5 cleanValidData
        !testData  = keepSplit 0.5 cleanTestData
        !trainBatches = makeBatch batchSize trainData
        !valBatches   = makeBatch batchSize validData
        !indexedTrainBatches = zip [1..] trainBatches
        !indexedValBatches   = zip [1..] valBatches

    initModel <- initialize modelSpec embeddingPath

    putStrLn "*** Training ***"
    (trainedModel, losses) <- trainLoop indexedTrainBatches indexedValBatches 1 0 (1/0) initModel []

    let trainLossList = [x | (x, _) <- losses]
        validLossList = [x | (_, x) <- losses]

    putStrLn "\n*** Testing ***"
    let h0 = zeros' [hiddenDimSize]
        predictedClasses = map (\(wordIds, _) ->
            let predVal = asValue (forwardRegression trainedModel h0 wordIds) :: Float
            in min 5 (max 0 (round predVal)) :: Int
            ) testData

        actualClasses = map (\(_, targetRating) -> round targetRating :: Int) testData

    putStrLn "*** Results ***"
    evaluate [0..5] actualClasses predictedClasses

    putStrLn "\n*** Saving learning curve and parameters... ***"
    drawLearningCurve imgPath "Learning Curve RNN" [("train", trainLossList), ("valid", validLossList)]

    saveParams (emb trainedModel) modelOutPath
