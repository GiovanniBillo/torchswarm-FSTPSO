## FST-PSO in Pytorch

To run a test with different benchmark problems, after activating the environment and from the main directory:
```
python3 tests.py
```
This runs the efficient vectorized version of the algorithm by default. In the files, the serial version is kept for didactic purposes but not supported. 

In order to use the fuzzy self-tuning version of the algorithm after [Nobile et al.](https://www.sciencedirect.com/science/article/abs/pii/S2210650216303534):
```
python3 tests.py --model fuzzy
```

To run the first inverse problem to recover the parameters of Lotka Volterra equations:

Standard PSO:
```
python3 lv.py 
```
Fuzzy Self-Tuning PSO:
```
python3 lv.py --model fuzzy 
```

Lastly, for the simplified OCx polynomial parameter recovery experiment:
This experiment needs 
Standard PSO:
```
python3 inverse_ocx.py 
```
Fuzzy Self-Tuning PSO:
```
python3 inverse_ocx.py --model fuzzy 
```


