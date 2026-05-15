{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}

{-|
Module      : LoadData
Description : Generic CSV data loading and preprocessing utility.
-}
module LoadData (loadFromCSV, standardize) where

import System.IO
import System.Exit (exitFailure)
import Data.Either (rights)
import Data.ByteString (ByteString, hGetSome, empty)
import Data.Csv.Incremental
import Data.Csv (FromRecord)
import Torch.Tensor (Tensor, asTensor)
import Torch (stdMeanDim, Dim(..), KeepDim(..))

-- | Standardizes a tensor by subtracting the mean and dividing by the standard deviation.
-- A small epsilon (1e-8) is added to prevent division by zero.
standardize :: Tensor -> Tensor
standardize xs =
    let (xStds, xMeans) = stdMeanDim (Dim 0) True KeepDim xs
    in (xs - xMeans) / (xStds + 1e-8)

-- | Streams data.
feed :: (ByteString -> Parser a) -> Handle -> IO (Parser a)
feed k csvFile = do
    hIsEOF csvFile >>= \case
        True  -> return $ k empty
        False -> k <$> hGetSome csvFile 4096

-- | Parses a CSV file into feature and target tensors using a custom extractor function.
-- The CSV file must contain a header row.
-- Fails and exits the program if the CSV format is invalid.
loadFromCSV :: forall a. (FromRecord a)
            => String                    -- ^ Path to the CSV file
            -> (a -> ([Float], [Float])) -- ^ Extractor mapping a record to (Features, Targets)
            -> IO (Tensor, Tensor)
loadFromCSV path extractor = do
    withFile path ReadMode $ \ csvFile -> do
        let loop !_ (Fail _ errMsg) = do
                putStrLn $ "Erreur de parsing: " ++ errMsg
                exitFailure

            loop acc (Many rs k) =
                loop (acc <> rs) =<< feed k csvFile

            loop acc (Done rs) = do
                let allRecords = rights (acc <> rs)
                    -- Decouple data extraction by relying on the injected extractor function
                    (xsList, ysList) = unzip (map extractor allRecords)

                    xTensor = asTensor xsList
                    yTensor = asTensor ysList

                    xStandardized = standardize xTensor

                return (xStandardized, yTensor)

        loop [] (decode HasHeader)
