{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Word2vec (main) where

import Data.Char (toLower)
import Codec.Binary.UTF8.String (encode, decode)
import GHC.Generics
import qualified Data.ByteString.Lazy as B
import Data.Word (Word8)
import qualified Data.Map.Strict as M
import Data.List (group, sort, sortOn)
import Data.Ord (Down(..))
import Data.List.Split (splitPlaces)
import Control.Monad (when)
import Torch
import Torch.Layer.MLP (MLPHypParams(..), MLPParams(..), mlpLayer, ActName(..))
import Torch.Control (makeBatch, mapAccumM, foldLoopM)
import ML.Exp.Chart (drawLearningCurve)
import Text.Printf (printf)
import System.IO (hFlush, stdout)
import System.Mem (performMajorGC)

batchSize :: Int
batchSize = 64

numEpoch :: Int
numEpoch = 10

learningRate :: Tensor
learningRate = 1e-2

dimention :: Int
dimention = 8

lossFn :: Tensor -> Tensor -> Tensor
lossFn y y' = let logProbs = logSoftmax (Dim 1) y'
    in nllLoss' (squeezeAll y) (squeezeAll logProbs)

textFilePath :: String
textFilePath = "Session6/data/review-texts.txt"
imgPath :: String
imgPath = "Session6/img/learning-curve.png"
modelPath :: String
modelPath = "Session6/data/sample_embedding.params"
wordLstPath :: String
wordLstPath = "Session6/data/sample_wordlst.txt"

unncessaryChars :: [String]
unncessaryChars = [".", "!", ",", ";", "?", "<", ">", "\\", "/", "(", ")", "\"", "'"]

stopWords :: [String]
stopWords = ["the", "a", "an", "is", "are", "was", "and", "of", "to", "in", "it", "this", "that", "br"]

data EmbeddingSpec = EmbeddingSpec {
    wordNum :: Int,
    wordDim :: Int
} deriving (Show, Eq, Generic)

data Embedding = Embedding {
    wordEmbedding :: Parameter
} deriving (Show, Generic, Parameterized)

data Model = Model {
    mlp :: MLPParams,
    embeddings :: Embedding
} deriving (Generic, Parameterized)


isUnncessaryChar :: Word8 -> Bool
isUnncessaryChar char = char `elem` map (head . encode) unncessaryChars

isStopWord :: B.ByteString -> Bool
isStopWord word = word `elem` map (B.pack . encode) stopWords

preprocess :: B.ByteString -> [[B.ByteString]]
preprocess texts = map (filter (not . isStopWord) . B.split (head $ encode " ")) textLines
  where
    lowerBytes = encode $ map toLower (decode $ B.unpack texts)
    filteredtexts = B.pack $ filter (not . isUnncessaryChar) lowerBytes
    textLines = B.split (head $ encode "\n") filteredtexts

wordToIndexFactory :: [B.ByteString] -> (B.ByteString -> Int)
wordToIndexFactory wordlst wrd = M.findWithDefault (length wordlst) wrd (M.fromList (zip wordlst [0.. length wordlst]))

toyEmbedding :: EmbeddingSpec -> Tensor
toyEmbedding EmbeddingSpec{..} = eye' wordNum wordDim

createLst :: [B.ByteString] -> Float -> [B.ByteString]
createLst allWords keepWordsRatio =
    let groupedWords = group (sort allWords)
        frequencies = map (\g -> (head g, length g)) groupedWords
        sortedWordsWithFreq = sortOn (Down . snd) frequencies
        sortedWords = map fst sortedWordsWithFreq
        totalUniqueWords = fromIntegral (length groupedWords) :: Float
        keepCount = Prelude.max 1 (round (totalUniqueWords * keepWordsRatio))
    in Prelude.take keepCount sortedWords

trainTestSplit :: (Float, Float) -> [e] -> ([e], [e])
trainTestSplit (trainRatio, testRatio) xs =
    let xLength = length xs
        trainSize = round (fromIntegral xLength * trainRatio)
        testSize = round (fromIntegral xLength * testRatio)
        endSize = xLength - trainSize
        resultList = splitPlaces [trainSize, testSize, endSize] xs
    in ((resultList !! 0), (resultList !! 1))

trainBatch :: [(Int, Int)] -- ^ Batch of (Features, Targets)
            -> (Model, Float)    -- ^ Model to train and accumulated training loss
            -> IO (Model, Float) -- ^ Model, accumulated training loss
trainBatch batch (currModel, totalLoss) = do
    let x = asTensor (map fst batch)
        y = asTensor (map snd batch)

        emb = embedding' (toDependent $ wordEmbedding $ embeddings currModel) x
        y' = mlpLayer (mlp currModel) emb
        batchLoss = lossFn y y'

    (newModel, _) <- runStep currModel GD batchLoss learningRate
    return (newModel, totalLoss + (asValue batchLoss :: Float))

valBatch :: [(Int, Int)] -- ^ Batch of (Features, Targets)
          -> (Model, Float)    -- ^ Model to evaluate and accumulated validation loss
          -> IO (Model, Float) -- ^ Model, accumulated validation loss
valBatch batch (currModel, totalLoss) = do
    let x = asTensor (map fst batch)
        y = asTensor (map snd batch)

        emb = embedding' (toDependent $ wordEmbedding $ embeddings currModel) x
        y' = mlpLayer (mlp currModel) emb
        batchLoss = lossFn y y'
    return (currModel, totalLoss + (asValue batchLoss :: Float))

processEpoch :: [[(Int, Int)]] -- ^ Training set containing (Features, Targets)
             -> [[(Int, Int)]] -- ^ Validation set containing (Features, Targets)
             -> Int                  -- ^ Current epoch number (used for logging)
             -> Model                -- ^ Current state of the model parameters
             -> IO (Model, (Float, Float)) -- ^ Return the new model parameters and training/validation losses
processEpoch trainBatches valBatches epoch model = do
    -- Train the model on the training set
    (newModel, totalTrainLoss) <- foldLoopM trainBatches (model, 0.0) trainBatch
    let avgTrainLoss = totalTrainLoss / fromIntegral (length trainBatches)

    -- Validate the new model on unseen data to monitor for overfitting
    (_, totalValLoss) <- foldLoopM valBatches (newModel, 0.0) valBatch
    let avgValLoss = totalValLoss / fromIntegral (length valBatches)

    -- Log progress periodically rather than every epoch to reduce standard output noise
    when (epoch `mod` 1 == 0) $ do
        printf "  Epoch %4d | Train Loss: %12.6e | Val Loss: %12.6e\n" epoch avgTrainLoss avgValLoss
        hFlush stdout

    performMajorGC

    return (newModel, (avgTrainLoss, avgValLoss))

main :: IO ()
main = do
    texts <- B.readFile textFilePath
    let allWords = concat $ preprocess texts
        wordlst = createLst allWords 0.01
        wordToIndex = wordToIndexFactory wordlst

        vocabSize = length wordlst + 1
        embeddingSpec = EmbeddingSpec {wordNum = vocabSize, wordDim = dimention}
    wordEmb <- makeIndependent $ toyEmbedding embeddingSpec

    putStrLn $ "Vocab Size: " ++ show vocabSize

    let emb = Embedding { wordEmbedding = wordEmb }
        hypParams = MLPHypParams (Device CPU 0) dimention [(128, Relu), (vocabSize, Id)]
    initMlp <- sample hypParams

    let initModel = Model { mlp = initMlp, embeddings = emb }

        idxes = map wordToIndex allWords
        dataset = zip (init idxes) (tail idxes)
        (trainData, validData) = trainTestSplit (0.01, 0.001) dataset
        trainBatches = makeBatch batchSize trainData
        valBatches = makeBatch batchSize validData

    putStrLn $ "Train Size: " ++ show (length trainBatches) ++ " Batches | Test Size: " ++ show (length valBatches) ++ " Batches"

    (trainedModel, lossesR) <- mapAccumM [1..numEpoch] initModel $ processEpoch trainBatches valBatches

    let losses    = reverse lossesR
        trainLoss = [x :: Float | (x, _) <- losses]
        validLoss = [x :: Float | (_, x) <- losses]

        trainedEmb = embeddings trainedModel

    drawLearningCurve imgPath "Learning Curve" [("train", trainLoss), ("valid", validLoss)]
    saveParams trainedEmb modelPath
    B.writeFile wordLstPath (B.intercalate (B.pack $ encode "\n") wordlst)

    return ()
