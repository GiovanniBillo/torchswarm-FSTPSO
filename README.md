## FST-PSO in Pytorch

To run a test with different benchmark problems, after activating the environment:
```
python3 tests.py
```
This will use the serial version of the normal algorithm by default. In order to use the more efficient vectorized version
```
python3 tests.py --mode parallel
```

In order to use the fuzzy self-tuning version of the algorithm
```
python3 tests.py --mode parallel --model fuzzy
```
