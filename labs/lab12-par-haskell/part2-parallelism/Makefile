hpar:
	ghc -dynamic -O2 --make hellopar.hs -threaded -rtsopts -o hpar
rpar:
	ghc -O2 rpar.hs -threaded
clean:
	rm hellopar.o rpar.o hellopar.hi rpar.hi rpar hpar
