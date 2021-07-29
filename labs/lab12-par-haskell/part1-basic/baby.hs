sumx :: Int -> Int -> Int
sumx _ 10 = 1
sumx 10 _ = 2
sumx x y = x+y

factorial :: Integer -> Integer  
factorial n = product [1..n]
-- addthree
addThree :: Int -> Int -> Int -> Int  
addThree x y z = x + y + z
-- comentarios, mi primera funcion
doubleMe x = x + x
-- otra funcion ejemplo
doubleUs x y = 2*x + 2*y
-- otra condicional
doubleSmallNumber x = if x > 100
then x
else x*2
