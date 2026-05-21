{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Word2vec (main) where

import Codec.Binary.UTF8.String (encode)
import GHC.Generics
import qualified Data.ByteString.Lazy as B
import Data.Word (Word8)
import qualified Data.Map.Strict as M
import Data.List (nub, group, sort, sortOn)
import Data.Ord (Down(..))
import Data.List.Split (splitPlaces)
import Control.Monad (when)
import Torch
import Torch.Layer.MLP (MLPHypParams(..), MLPParams(..), mlpLayer, ActName(..))
import Torch.Control (makeBatch, trainLoop)
import ML.Exp.Chart (drawLearningCurve)
import Text.Printf (printf)
import System.IO (hFlush, stdout)

batchSize :: Int
batchSize = 64

numEpoch :: Int
numEpoch = 500

learningRate :: Tensor
learningRate = 1e-2

dimention :: Int
dimention = 16

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
unncessaryChars = [".", "!"]

stopWords :: [String]
stopWords = ["the", "a", "an", "is", "are", "was", "and", "of", "to", "in", "it", "this", "that"]

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
preprocess texts =
    let space = head (encode " ")
        newline = head (encode "\n")
        filteredChars = B.pack $ filter (not . isUnncessaryChar) (B.unpack texts)
        textLines = B.split newline filteredChars
    in map (filter (not . isStopWord) . B.split space) textLines

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

trainTestSplit :: Float -> [e] -> ([e], [e])
trainTestSplit trainRatio xs =
    let xLength = fromIntegral (length xs)
        trainSize = round (fromIntegral xLength * trainRatio)
        testSize  = xLength - trainSize
        resultList = splitPlaces [trainSize, testSize] xs
    in ((resultList !! 0), (resultList !! 1))

trainStep ::
    Int ->
    [(Int, Int)] ->
    Model ->
    IO (Model, (Float, Float))
trainStep epoch batch model = do
    let xTrain = asTensor (map fst batch)
        yTrain = asTensor (map snd batch)

    let embTrain = embedding' (toDependent $ wordEmbedding $ embeddings model) xTrain
        y'Train = mlpLayer (mlp model) embTrain
        trainLoss = lossFn yTrain y'Train
        trainLossValue = asValue trainLoss :: Float

    (newModel, _) <- runStep model GD trainLoss learningRate

    when (epoch `mod` 50 == 0) $ do
        printf "  Epoch %4d | Train Loss: %12.6e | Val Loss: %12.6e\n" epoch trainLossValue (0 :: Float)
        hFlush stdout

    return (newModel, (trainLossValue, 0))

main :: IO ()
main = do
    texts <- B.readFile textFilePath
    let allWords = concat $ preprocess texts
        wordlst = createLst allWords 0.05
        wordToIndex = wordToIndexFactory wordlst

        vocabSize = length wordlst + 1
        embeddingSpec = EmbeddingSpec {wordNum = vocabSize, wordDim = dimention}
    wordEmb <- makeIndependent $ toyEmbedding embeddingSpec

    putStrLn $ "" ++ show vocabSize

    let emb = Embedding { wordEmbedding = wordEmb }
        hypParams = MLPHypParams (Device CPU 0) dimention [(128, Relu), (vocabSize, Id)]
    initMlp <- sample hypParams

    let initialModel = Model { mlp = initMlp, embeddings = emb }

        idxes = map wordToIndex allWords
        dataset = zip (init idxes) (tail idxes)
        (trainData, testData) = trainTestSplit 0.95 dataset
        batches = makeBatch batchSize trainData

    (trainedModel, allLosses) <- trainLoop [1..numEpoch] batches initialModel trainStep

    let losses    = reverse allLosses
        trainLoss = [x :: Float | (x, _) <- losses]
        validLoss = [x :: Float | (_, x) <- losses]

        trainedEmb = embeddings trainedModel

    drawLearningCurve imgPath "Learning Curve" [("train", trainLoss)]
    saveParams trainedEmb modelPath
    B.writeFile wordLstPath (B.intercalate (B.pack $ encode "\n") wordlst)

    return ()
