import Control.Parallel
import Control.Parallel.Strategies
import Control.Exception
import Data.Time.Clock
import Text.Printf
import System.Environment

-- <<fib
fib :: Integer -> Integer
fib 0 = 1
fib 1 = 1
fib n = fib (n-1) + fib (n-2)
-- >>

-- <<main
main = do
  [n] <- getArgs
  let test = [test1,test2,test3,test4] !! (read n - 1)
  t0 <- getCurrentTime
  r <- evaluate (runEval test)
  printTimeSince t0
  print r
  printTimeSince t0
-- >>

-- <<test1
test1 = do
  x <- rpar (fib 36)
  y <- rpar (fib 35)
  return (x,y)
-- >>

-- <<test2
test2 = do
  x <- rpar (fib 36)
  y <- rseq (fib 35)
  return (x,y)
-- >>

-- <<test3
test3 = do
  x <- rpar (fib 36)
  y <- rseq (fib 35)
  rseq x
  return (x,y)
-- >>

-- <<test4
test4 = do
  x1 <- rpar (fib 35)
  x2 <- rpar (fib 34)
  x3 <- rpar (fib 33)
  x4 <- rpar (fib 34)
  rseq x1
  rseq x2
  rseq x3
  rseq x4
  return (x1+x2, x3+x4)
-- >>

printTimeSince t0 = do
  t1 <- getCurrentTime
  printf "time: %.2fs\n" (realToFrac (diffUTCTime t1 t0) :: Double)
