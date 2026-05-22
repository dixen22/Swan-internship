{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE StandaloneDeriving #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Eval (main) where

import Codec.Binary.UTF8.String (encode)
import GHC.Generics
import qualified Data.ByteString.Lazy as B
import qualified Data.Map.Strict as M
import Data.Maybe (catMaybes)
import Text.Read (readMaybe)

import Torch
import Torch.Serialize (loadParams)

modelPath :: String
modelPath = "Session6/data/sample_embedding.params"
wordLstPath :: String
wordLstPath = "Session6/data/sample_wordlst.txt"
stsFilePath :: String
stsFilePath = "Session6/data/answer-answer.test.tsv"

dimention :: Int
dimention = 8

data Embedding = Embedding {
    wordEmbedding :: Parameter
} deriving (Generic, Parameterized)

preprocess :: B.ByteString -> [[B.ByteString]]
preprocess texts = map (B.split (head $ encode " ")) textLines
  where
    textLines = B.split (head $ encode "\n") texts

wordToIndexFactory :: [B.ByteString] -> (B.ByteString -> Int)
wordToIndexFactory wordlst wrd =
    M.findWithDefault (length wordlst) wrd (M.fromList (zip wordlst [0.. length wordlst]))

getSentenceVector :: Embedding -> (B.ByteString -> Int) -> String -> Tensor
getSentenceVector loadedEmb wordToIndexFunc sentence =
    let sentenceBs = B.pack (encode sentence)
        wordsList = concat (preprocess sentenceBs)

        idxs = if null wordsList
               then [wordToIndexFunc (B.pack [])]
               else map wordToIndexFunc wordsList

        idxTensor = asTensor idxs
        wordVecs = embedding' (toDependent $ wordEmbedding loadedEmb) idxTensor
        sentenceVec = meanDim (Dim 0) KeepDim Float wordVecs
    in sentenceVec

splitTabs :: String -> [String]
splitTabs "" = [""]
splitTabs (c:cs)
    | c == '\t' = "" : rest
    | otherwise = (c : head rest) : tail rest
    where rest = splitTabs cs

cosToScore :: Float -> Int
cosToScore cosSim
    | cosSim < -0.6 = 0
    | cosSim < -0.2 = 1
    | cosSim <  0.2 = 2
    | cosSim <  0.6 = 3
    | cosSim <  0.8 = 4
    | otherwise     = 5

processLine :: Embedding -> (B.ByteString -> Int) -> String -> IO (Maybe (Int, Int))
processLine loadedEmb wordToIndexFunc line = case splitTabs line of
    (labelStr : sent1 : sent2 : _) ->
        case readMaybe labelStr :: Maybe Float of
            Just humanScoreFloat -> do
                let humanScore = round humanScoreFloat

                    vec1 = getSentenceVector loadedEmb wordToIndexFunc sent1
                    vec2 = getSentenceVector loadedEmb wordToIndexFunc sent2

                    simTensor = cosineSimilarity' vec1 vec2
                    simVal = asValue simTensor :: Float
                    modelScore = cosToScore simVal

                return $ Just (humanScore, modelScore)
            Nothing -> return Nothing
    _ -> return Nothing

evaluateSTS :: Embedding -> (B.ByteString -> Int) -> FilePath -> IO ()
evaluateSTS loadedEmb wordToIndexFunc tsvPath = do
    content <- readFile tsvPath
    let lns = lines content

    processedMaybes <- mapM (processLine loadedEmb wordToIndexFunc) lns
    let results = catMaybes processedMaybes

    let totalPairs = length results
        exactMatches = length (filter (\(humain, modele) -> humain == modele) results)
        closeMatches = length (filter (\(humain, modele) -> Prelude.abs (humain - modele) <= 1) results)

        calcPercent part = (fromIntegral part / fromIntegral totalPairs) * 100 :: Float

    putStrLn "Results"
    putStrLn $ "  Pairs evaluated   : " ++ show totalPairs
    putStrLn $ "  Exact predictions : " ++ show exactMatches ++ " (" ++ show (calcPercent exactMatches) ++ "%)"
    putStrLn $ "  Close predictions : " ++ show closeMatches ++ " (" ++ show (calcPercent closeMatches) ++ "%)"

main :: IO ()
main = do
    wordLstContent <- B.readFile wordLstPath
    let wordlst = B.split (head $ encode "\n") wordLstContent
        wordToIndex = wordToIndexFactory wordlst
        vocabSize = length wordlst + 1

    initWordEmb <- makeIndependent $ zeros' [1]
    let initEmb = Embedding {wordEmbedding = initWordEmb}
    loadedEmb <- loadParams initEmb modelPath

    evaluateSTS loadedEmb wordToIndex stsFilePath
