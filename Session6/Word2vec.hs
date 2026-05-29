{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}

-- |
-- Module      : Word2vec
-- Description : An Word2Vec implementation using Hasktorch.
--
-- This module implements a basic word embedding model with a training loop
-- that includes early stopping and validation checks.
module Word2vec (main) where

import Data.Char (toLower)
import Codec.Binary.UTF8.String (encode)
import GHC.Generics
import qualified Data.ByteString.Lazy as B
import qualified Data.ByteString.Lazy.Char8 as B8
import Data.Word (Word8)
import qualified Data.HashMap.Strict as HM
import Data.Maybe (fromMaybe)
import Data.List (sortOn, foldl')
import Data.Ord (Down(..))
import Data.List.Split (splitPlaces)
import Control.Monad (when)
import Torch
import Torch.Layer.MLP (MLPHypParams(..), MLPParams(..), mlpLayer, ActName(..))
import Torch.Control (makeBatch, foldLoopM)
import ML.Exp.Chart (drawLearningCurve)
import Text.Printf (printf)
import System.IO (hFlush, stdout)
import System.Mem (performMinorGC)

dimension :: Int
dimension = 64

batchSize :: Int
batchSize = 2048

maxEpoch :: Int
maxEpoch = 50

learningRate :: Tensor
learningRate = 1e-1

patience :: Int
patience = 2

delta :: Float
delta = 5e-3

textFilePath, imgPath, modelPath, wordLstPath :: String
textFilePath = "Session6/data/review-texts.txt"
imgPath = "Session6/img/learning-curve.png"
modelPath = "Session6/data/sample_embedding.params"
wordLstPath = "Session6/data/sample_wordlst.txt"

unnecessaryChars, stopWords :: [String]
unnecessaryChars = [".", "!", ",", ";", "?", "<", ">", "\\", "/", "(", ")", "\"", "'"]
stopWords = ["the", "a", "an", "is", "are", "was", "and", "of", "to", "in", "it", "this", "that", "br", " "]

-- | Specification for the embedding layer.
-- Optimization: Added "!" (BangPatterns) to enforce strict evaluation and prevent space leaks.
data EmbeddingSpec = EmbeddingSpec {
    wordNum :: !Int,
    wordDim :: !Int
} deriving (Show, Eq, Generic)

-- | The embedding layer containing the trainable parameters.
data Embedding = Embedding {
    wordEmbedding :: !Parameter
} deriving (Show, Generic, Parameterized)

-- | The complete model composed of an MLP and an Embedding layer.
data Model = Model {
    mlp :: !MLPParams,
    embeddings :: !Embedding
} deriving (Generic, Parameterized)

-- | Checks if a character is considered unnecessary punctuation.
isUnnecessaryChar :: Word8 -> Bool
isUnnecessaryChar char = char `elem` map (head . encode) unnecessaryChars

-- | Checks if a word is in the predefined stop words list.
isStopWord :: B.ByteString -> Bool
isStopWord word = word `elem` map (B.pack . encode) stopWords

-- | Preprocesses the raw text by lowercasing, removing punctuation,
-- filtering stop words, and splitting by spaces/newlines.
preprocess :: B.ByteString -> [[B.ByteString]]
preprocess texts = map (filter (not . isStopWord) . B.split 32) textLines
  where
    lowerBytes = B8.map toLower texts
    filteredtexts = B.filter (not . isUnnecessaryChar) lowerBytes
    textLines = B.split 10 filteredtexts

-- | Creates a function that maps a word (ByteString) to its corresponding integer index.
-- Unknown words are assigned the index equal to the vocabulary size.
wordToIndexFactory :: [B.ByteString] -> (B.ByteString -> Int)
wordToIndexFactory wordlst wrd = fromMaybe (length wordlst) (HM.lookup wrd dict)
    where
        !dict = HM.fromList (zip wordlst [0..])

-- | Initializes an identity matrix as a toy embedding for demonstration.
toyEmbedding :: EmbeddingSpec -> Generator -> Tensor
toyEmbedding EmbeddingSpec{..} gen =
    let (t, _) = randn' [wordNum, wordDim] gen
    in mulScalar (0.01 :: Float) t

-- | Creates a vocabulary list keeping only the most frequent words based on a ratio.
createLst :: [B.ByteString] -> Float -> [B.ByteString]
createLst allWords keepWordsRatio =
    let freqMap = foldl' (\acc w -> HM.insertWith (+) w (1 :: Int) acc) HM.empty allWords
        frequencies = HM.toList freqMap
        sortedWordsWithFreq = sortOn (Down . snd) frequencies
        sortedWords = map fst sortedWordsWithFreq
        totalUniqueWords = fromIntegral (HM.size freqMap) :: Float
        keepCount = Prelude.max 1 (round (totalUniqueWords * keepWordsRatio))
    in Prelude.take keepCount sortedWords

-- | Splits a dataset into training and validation sets according to the provided ratios.
trainTestSplit :: (Float, Float) -> [e] -> ([e], [e])
trainTestSplit (trainRatio, testRatio) xs =
    let xLength = length xs
        trainSize = round (fromIntegral xLength * trainRatio)
        testSize = round (fromIntegral xLength * testRatio)
        endSize = xLength - trainSize
        resultList = splitPlaces [trainSize, testSize, endSize] xs
    in (head resultList, resultList !! 1)

-- | Converts a list of (Input, Target) index pairs into a tuple of Tensors.
toBatchTensor :: [(Int, Int)] -> (Tensor, Tensor)
toBatchTensor batch = (asTensor (map fst batch), asTensor (map snd batch))

-- | Computes the Negative Log-Likelihood loss.
lossFn :: Tensor -> Tensor -> Tensor
lossFn y y' = let logProbs = logSoftmax (Dim 1) y'
    in nllLoss' (squeezeAll y) (squeezeAll logProbs)

-- | Trains the model on a single batch and accumulates the training loss.
trainBatch :: Int
           -> (Int, (Tensor, Tensor))
           -> (Model, Float)
           -> IO (Model, Float)
trainBatch totalBatches (batchIdx, (x, y)) (currModel, !totalLoss) = do
    let emb = embedding' (toDependent $ wordEmbedding $ embeddings currModel) x
        y' = mlpLayer (mlp currModel) emb
        batchLoss = lossFn y y'

    (newModel, _) <- runStep currModel GD batchLoss learningRate

    when (batchIdx `mod` 50 == 0 || batchIdx == totalBatches) $ do
        printf "\r\ESC[K  Training %d/%d" batchIdx totalBatches
        hFlush stdout

    performMinorGC
    return (newModel, totalLoss + (asValue batchLoss :: Float))

-- | Evaluates the model on a single validation batch and accumulates the validation loss.
valBatch :: Int
         -> (Int, (Tensor, Tensor))
         -> (Model, Float)
         -> IO (Model, Float)
valBatch totalBatches (batchIdx, (x, y)) (currModel, !totalLoss) = do
    let emb = embedding' (toDependent $ wordEmbedding $ embeddings currModel) x
        y' = mlpLayer (mlp currModel) emb
        batchLoss = lossFn y y'

    when (batchIdx `mod` 50 == 0 || batchIdx == totalBatches) $ do
        printf "\r\ESC[K  Validation %d/%d" batchIdx totalBatches
        hFlush stdout

    performMinorGC
    return (currModel, totalLoss + (asValue batchLoss :: Float))

-- | Processes a full epoch: runs training and validation over all batches.
processEpoch :: [(Int, (Tensor, Tensor))] -- ^ Training batches
             -> [(Int, (Tensor, Tensor))] -- ^ Validation batches
             -> Int                       -- ^ Current epoch number
             -> Model                     -- ^ Current model state
             -> IO (Model, (Float, Float))
processEpoch trainBatches valBatches epoch model = do
    putStrLn $ "Epoch " ++ show epoch

    let totalTrainBatches = length trainBatches
        totalValBatches = length valBatches

    -- Train the model
    (newModel, totalTrainLoss) <- foldLoopM trainBatches (model, 0.0) (trainBatch totalTrainBatches)
    let avgTrainLoss = totalTrainLoss / fromIntegral totalTrainBatches

    putStrLn ""

    -- Validate the model
    (_, totalValLoss) <- foldLoopM valBatches (newModel, 0.0) (valBatch totalValBatches)
    let avgValLoss = totalValLoss / fromIntegral totalValBatches

    -- Log progress
    printf "\n  Train Loss: %12.6e | Val Loss: %12.6e\n" avgTrainLoss avgValLoss
    hFlush stdout

    return (newModel, (avgTrainLoss, avgValLoss))

-- | Main training loop with early stopping mechanism.
trainLoop :: [(Int, (Tensor, Tensor))] -- ^ Training batches
          -> [(Int, (Tensor, Tensor))] -- ^ Validation batches
          -> Int                       -- ^ Current Epoch
          -> Int                       -- ^ Patience counter
          -> Float                     -- ^ Best Validation Loss
          -> Model                     -- ^ Current Model
          -> [(Float, Float)]          -- ^ Accumulated Losses
          -> IO (Model, [(Float, Float)])
trainLoop trainBatches valBatches epoch patienceCounter bestValLoss model losses
    | epoch > maxEpoch || patienceCounter >= patience = do
        return (model, reverse losses)
    | otherwise = do
        (newModel, (trainLoss, valLoss)) <- processEpoch trainBatches valBatches epoch model
        let !newLosses = (trainLoss, valLoss) : losses

        let nextBestLoss = if valLoss < bestValLoss
            then valLoss
            else bestValLoss

        let nextPatience = if valLoss < bestValLoss - delta
            then 0
            else patienceCounter + 1

        trainLoop trainBatches valBatches (epoch + 1) nextPatience nextBestLoss newModel newLosses

main :: IO ()
main = do
    -- Load and preprocess text
    rawText <- B.readFile textFilePath
    let processedLines = preprocess rawText
        allWords = concat processedLines
        wordlst = createLst allWords 0.15
        vocabSize = length wordlst + 1
        wordToIndex = wordToIndexFactory wordlst
        embeddingSpec = EmbeddingSpec vocabSize dimension

    putStrLn $ "Vocab Size      : " ++ show vocabSize
    putStrLn $ "Embedding Dim   : " ++ show dimension

    let !idxes = map wordToIndex allWords
        !dataset = zip (init idxes) (tail idxes)
        (!trainData, !validData) = trainTestSplit (0.1, 0.01 ) dataset

        !trainBatches = map toBatchTensor (makeBatch batchSize trainData)
        !valBatches   = map toBatchTensor (makeBatch batchSize validData)

        !indexedTrainBatches = zip [1..] trainBatches
        !indexedValBatches   = zip [1..] valBatches

    gen <- mkGenerator (Device CPU 0) 42
    wordEmb <- makeIndependent $ toyEmbedding embeddingSpec gen

    let emb = Embedding { wordEmbedding = wordEmb }
        hypParams = MLPHypParams (Device CPU 0) dimension [(vocabSize, Id)]

    initMlp <- sample hypParams

    let initModel = Model { mlp = initMlp, embeddings = emb }

    (trainedModel, losses) <- trainLoop indexedTrainBatches indexedValBatches 1 0 (1/0) initModel []

    let trainLoss = [x :: Float | (x, _) <- losses]
        validLoss = [x :: Float | (_, x) <- losses]
        trainedEmb = embeddings trainedModel

    putStrLn "\nSaving model and learning curve"
    drawLearningCurve imgPath "Learning Curve" [("train", trainLoss), ("valid", validLoss)]
    saveParams trainedEmb modelPath
    B.writeFile wordLstPath (B.intercalate (B.pack $ encode "\n") wordlst)

    return ()
