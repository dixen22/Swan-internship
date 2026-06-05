{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ScopedTypeVariables #-}

module CnnText (main) where -- Correction du Warning "missing-export-lists"

import Codec.Binary.UTF8.String (encode)
import Data.Aeson (FromJSON(..), ToJSON(..), eitherDecode)
import qualified Data.ByteString.Lazy as B
import qualified Data.ByteString.Internal as B (c2w)
import qualified Data.ByteString.Lazy.Char8 as B8
import qualified Data.Map.Strict as M
import GHC.Generics
import Data.Int (Int64)
import Data.Char (toLower, isAlphaNum)
import Data.List.Split (splitPlaces)
import Control.Monad (when, foldM)
import Text.Printf (printf)
import System.IO (hFlush, stdout)
import System.Mem (performMinorGC)

import Torch.NN hiding (forward)
import Torch.Tensor
import Torch.Functional hiding (min, max, take, repeat, mseLoss)
import Torch.Serialize (loadParams, saveParams)
import Torch.TensorFactories (randnIO', zeros')
import Torch.Autograd (makeIndependent, toDependent)
import Torch.Optim (GD(..), runStep)
import Torch.Control (makeBatch, foldLoopM)
import ML.Exp.Chart (drawLearningCurve)
import qualified Torch as T
import Evaluation (evaluate)

-- LA CORRECTION : On force l'aléatoire pour éviter le crash "index out of range"
embeddingIsRandom :: Bool
embeddingIsRandom = True

seq_len :: Int
seq_len = 35

embSize :: Int
embSize = 128

batchSize :: Int
batchSize = 32

maxEpoch :: Int
maxEpoch = 15

learningRate :: Tensor
learningRate = asTensor (0.001 :: Float)

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
modelOutPath = "Session8/data/cnn_model.params"
imgPath      = "Session8/img/learning-curve-cnn.png"

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

type Batch = [([Int64], Float)]

data CnnTextBBSpec = CnnTextBBSpec
  { conv1 :: Conv1dSpec,
    conv2 :: Conv1dSpec,
    conv3 :: Conv1dSpec,
    conv4 :: Conv1dSpec,
    fc    :: LinearSpec
  } deriving (Show, Eq)

cnnTextBackBoneSpec :: CnnTextBBSpec
cnnTextBackBoneSpec =
  CnnTextBBSpec
    (Conv1dSpec embSize outSize 2)
    (Conv1dSpec embSize outSize 3)
    (Conv1dSpec embSize outSize 4)
    (Conv1dSpec embSize outSize 5)
    (LinearSpec inFc 1)
  where
    outSize = 32
    inFc = 4 * outSize -- 4 branches de convolution

data CnnTextBB = CnnTextBB
  { c1 :: Conv1d,
    c2 :: Conv1d,
    c3 :: Conv1d,
    c4 :: Conv1d,
    l1 :: Linear
  } deriving (Generic, Show, Parameterized)

instance Randomizable CnnTextBBSpec CnnTextBB where
  sample CnnTextBBSpec {..} =
    CnnTextBB
      <$> sample conv1
      <*> sample conv2
      <*> sample conv3
      <*> sample conv4
      <*> sample fc

data Embedding = Embedding {
  wordEmbedding :: Parameter
} deriving (Show, Generic, Parameterized)

data ModelSpec = ModelSpec {
  wordNum :: Int,
  wordDim :: Int
} deriving (Show, Eq, Generic)

data Model = Model {
  emb :: Embedding,
  cnn :: CnnTextBB
} deriving (Show, Generic, Parameterized)

instance Randomizable ModelSpec Model where
  sample ModelSpec {..} =
    Model
      <$> (Embedding <$> (makeIndependent =<< randnIO' [wordNum, wordDim]))
      <*> sample cnnTextBackBoneSpec

initialize :: ModelSpec -> FilePath -> IO Model
initialize modelSpec embPath = do
  randomizedModel <- sample modelSpec
  if embeddingIsRandom
    then return randomizedModel
    else do
      loadedEmb <- loadParams (emb randomizedModel) embPath
      return randomizedModel { emb = loadedEmb }

convLayer :: Conv1d -> Int -> Tensor -> Tensor
convLayer conv kernel input =
  squeezeDim 2
    . maxPool1d poolSize 1 0 1 Floor
    . relu
    . conv1dForward conv 1 0
    $ input
  where
    poolSize = seq_len - kernel + 1

forward :: Model -> Bool -> Tensor -> IO Tensor
forward Model{..} isTrain input = do
  let wEmb = toDependent (wordEmbedding emb)
      x = embedding' wEmb input
      x_t = transpose (Dim 1) (Dim 2) x
      x1 = convLayer (c1 cnn) 2 x_t
      x2 = convLayer (c2 cnn) 3 x_t
      x3 = convLayer (c3 cnn) 4 x_t
      x4 = convLayer (c4 cnn) 5 x_t

      combined = cat (Dim 1) [x1, x2, x3, x4]
      linOut = linear (l1 cnn) combined

  droppedOut <- dropout 0.25 isTrain linOut
  return $ squeezeAll (sigmoid droppedOut)

mseLoss :: Tensor -> Tensor -> Tensor
mseLoss prediction target = T.mean ((prediction - target) * (prediction - target))

calcBatchLoss :: Model -> Bool -> Batch -> IO Tensor
calcBatchLoss model isTrain batch = do
    let zeroLoss = zeros' [1]
        batchSizeTensor = asTensor [fromIntegral (length batch) :: Float]

    totalLoss <- foldM (\accLoss (wordIds, targetRating) -> do
         let inputTensor = reshape [1, seq_len] (asTensor wordIds)
         prediction <- forward model isTrain inputTensor
         let target = asTensor [targetRating]
         return $ accLoss + mseLoss prediction target
         ) zeroLoss batch
    return $ totalLoss / batchSizeTensor

trainBatch :: Int -> (Int, Batch) -> (Model, Float) -> IO (Model, Float)
trainBatch totalBatches (batchIdx, batch) (currModel, !totalLoss) = do
    batchLoss <- calcBatchLoss currModel True batch
    (newModel, _) <- runStep currModel GD batchLoss learningRate

    when (batchIdx `mod` 50 == 0 || batchIdx == totalBatches) $ do
        printf "\r\ESC[K  Training %d/%d" batchIdx totalBatches
        hFlush stdout

    performMinorGC
    return (newModel, totalLoss + (asValue batchLoss :: Float))

valBatch :: Int -> (Int, Batch) -> (Model, Float) -> IO (Model, Float)
valBatch totalBatches (batchIdx, batch) (currModel, !totalLoss) = do
    batchLoss <- calcBatchLoss currModel False batch

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

padOrTruncate :: Int -> [Int64] -> [Int64]
padOrTruncate n xs = take n (xs ++ repeat 0)

cleanDataset :: (B.ByteString -> Int64) -> [AmazonReview] -> [([Int64], Float)]
cleanDataset wordToIndex reviews =
  let validReviews = filter (not . null . preprocess . B8.pack . text) reviews
  in map (\r ->
    let tokens = concat $ preprocess (B8.pack $ text r)
        ids = padOrTruncate seq_len (map wordToIndex tokens)
    in (ids, rating r)) validReviews

main :: IO ()
main = do
  trainData' <- loadDataset trainReviewPath
  validData' <- loadDataset validReviewPath
  testData'  <- loadDataset testReviewPath

  wordLst <- fmap (B.split (head $ encode "\n")) (B.readFile wordLstPath)
  let wordToIndex = wordToIndexFactory wordLst
      totalWords  = length wordLst + 1
      modelSpec = ModelSpec { wordDim = embSize, wordNum = totalWords }

  let cleanTrainData = cleanDataset wordToIndex trainData'
      cleanValidData = cleanDataset wordToIndex validData'
      cleanTestData  = cleanDataset wordToIndex testData'

  let !trainData = keepSplit 0.05 cleanTrainData
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

  predictedClasses <- mapM (\(wordIds, _) -> do
          let inputTensor = reshape [1, seq_len] (asTensor wordIds)
          predVal <- asValue <$> forward trainedModel False inputTensor
          return $ (min 5 (max 0 (round (predVal :: Float))) :: Int)
      ) testData

  let actualClasses = map (\(_, targetRating) -> round targetRating :: Int) testData

  putStrLn "*** Results ***"
  evaluate [0..5] actualClasses predictedClasses

  putStrLn "\n*** Saving learning curve and parameters... ***"
  drawLearningCurve imgPath "Learning Curve CNN" [("train", trainLossList), ("valid", validLossList)]

  saveParams (emb trainedModel) modelOutPath
