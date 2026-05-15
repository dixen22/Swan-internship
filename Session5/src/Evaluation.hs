{-|
Module      : Evaluation
Description : Standard classification metrics and reporting tools.

This module provides functions to evaluate the performance of classification models.
It calculates standard metrics (Precision, Recall, F1, Accuracy) and generates
a comprehensive text report similar to scikit-learn's `classification_report`.

Assumption: Predictions and actual targets are represented as lists of discrete integers (classes).
-}
module Evaluation(recallScore, precisionScore, f1Score, f1Score', confusionMatrix, accuracy, evaluate, main) where

import Text.Printf (printf)

-- | Calculates the Recall (True Positive Rate) for a specific class.
-- Useful when the cost of False Negatives is high (e.g., medical diagnosis).
-- Returns NaN if the class does not appear in the actual dataset (division by zero).
recallScore :: Int -> [Int] -> [Int] -> Double
recallScore classNum actual predicted =
    let actuPos = filter (\(a, _) -> a == classNum) $ zip actual predicted
        truePos = filter (\(a, p) -> a == p) actuPos
    in fromIntegral (length truePos) / fromIntegral (length actuPos)

-- | Calculates the Precision (Positive Predictive Value) for a specific class.
-- Useful when the cost of False Positives is high (e.g., spam filtering).
-- Returns NaN if the model never predicted this class (division by zero).
precisionScore :: Int -> [Int] -> [Int] -> Double
precisionScore classNum actual predicted =
    let predPos = filter (\(_, p) -> p == classNum) $ zip actual predicted
        truePos = filter (\(a, p) -> a == p) predPos
    in fromIntegral (length truePos) / fromIntegral (length predPos)

-- | Computes the harmonic mean of precision and recall.
-- F1 Score is preferred over accuracy for imbalanced datasets because it punishes extreme values.
f1Score :: Double -> Double -> Double
f1Score recall precision = 2 * ((recall * precision) / (recall + precision))

-- | Convenience wrapper to compute the F1 Score directly from raw predictions.
f1Score' :: Int -> [Int] -> [Int] -> Double
f1Score' classNum actual predicted =
    let recall = recallScore classNum actual predicted
        precision = precisionScore classNum actual predicted
    in f1Score recall precision

-- | Generates a 2D confusion matrix to visualize where the model is making errors.
-- Rows represent the Actual classes, Columns represent the Predicted classes.
confusionMatrix :: [Int] -> [Int] -> [Int] -> [[Int]]
confusionMatrix classes actual predicted =
    let d = zip actual predicted
    in map (\classI -> [length $ filter (\(a, p) -> a == classI && p == classJ) d | classJ <- classes]) classes

printMatrix :: [Int] -> [[Int]] -> IO ()
printMatrix classes matrix = do
    let sepLine = "+" ++ replicate 10 '-' ++ concatMap (const "+--------") classes ++ "+"

    -- Affichage de l'en-tête du tableau (colonnes des prédictions)
    putStrLn sepLine
    putStr "|          |"
    mapM_ (printf " Pred %-2d|") classes
    putStrLn ""
    putStrLn sepLine

    -- Affichage des lignes de données (valeurs réelles et comptes)
    let printMatrixRow (actualLabel, counts) = do
            printf "| Actual %-2d|" actualLabel
            mapM_ (printf " %6d |") counts
            putStrLn ""
            putStrLn sepLine

    mapM_ printMatrixRow $ zip classes matrix

-- | Calculates the overall percentage of correct predictions.
accuracy :: [Int] -> [Int] -> Double
accuracy actual predicted =
    let truePred = filter (\(a, p) -> a == p) $ zip actual predicted
    in fromIntegral (length truePred) / fromIntegral (length actual)

-- | Unweighted mean of scores across all classes.
-- Treats all classes equally, regardless of their support (frequency) in the dataset.
macroAvg :: [Double] -> Double
macroAvg scores = (sum scores) / fromIntegral (length scores)

-- | Mean of scores, weighted by the number of true instances (support) for each class.
-- Accounts for class imbalance when evaluating overall model performance.
weightedAvg :: [Int] -> Int -> [Double] -> Double
weightedAvg supports totalSupport scores =
    let weight = map (\s -> fromIntegral s / fromIntegral totalSupport) supports
        weightedScores = map (\(w, s) -> w * s) $ zip weight scores
    in sum weightedScores

-- | Prints a comprehensive text report showing main classification metrics.
-- This function aggregates precision, recall, F1, and support per class,
-- followed by global averages and the confusion matrix.
evaluate :: [Int] -- ^ List of all possible unique classes
         -> [Int] -- ^ Ground truth (correct) target values
         -> [Int] -- ^ Estimated targets as returned by a classifier
         -> IO ()
evaluate classes actual predicted = do
    let supports   = map (\c -> length $ filter (== c) actual) classes
        recalls    = map (\c -> recallScore c actual predicted) classes
        precisions = map (\c -> precisionScore c actual predicted) classes
        f1Scores   = map (\(r, p) -> f1Score r p) $ zip recalls precisions

        macroPrec  = macroAvg precisions
        macroRec   = macroAvg recalls
        macroF1    = macroAvg f1Scores

        totalSupport = length actual
        weightedPrec = weightedAvg supports totalSupport precisions
        weightedRec  = weightedAvg supports totalSupport recalls
        weightedF1   = weightedAvg supports totalSupport f1Scores

        accu   = accuracy actual predicted
        matrix = confusionMatrix classes actual predicted

    putStrLn "             precision    recall    f1-score    support"
    putStrLn "             ------------------------------------------"

    let formatRow i = printf "    Class %2d  %8.2f  %8.2f    %8.2f       %4d\n"
                             (classes !! i) (precisions !! i) (recalls !! i) (f1Scores !! i) (supports !! i)
    mapM_ formatRow [0 .. length classes - 1]

    putStrLn "             ------------------------------------------"
    printf "    accuracy                        %8.2f       %4d\n" accu totalSupport
    printf "   macro avg  %8.2f  %8.2f    %8.2f       %4d\n" macroPrec macroRec macroF1 totalSupport
    printf "weighted avg  %8.2f  %8.2f    %8.2f       %4d\n" weightedPrec weightedRec weightedF1 totalSupport

    putStrLn "\nConfusion Matrix"
    printMatrix classes matrix

-- | Simple test harness to verify the evaluation logic with dummy data.
main :: IO ()
main = do
    let c = [1, 2, 3]
        a = [1, 1, 1, 2, 3, 3, 3, 3, 3, 3]
        p = [1, 2, 1, 2, 3, 1, 3, 2, 3, 2]

    evaluate c a p
