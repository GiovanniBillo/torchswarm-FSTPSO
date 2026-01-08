## FST-PSO in Pytorch

To run a test with different benchmark problems (e.g Ackley, Rastrigin, Griewank...), after activating the environment and from the main directory:
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

The final experiment addresses a simplified inverse problem inspired by **ocean-colour remote sensing**, focusing on the recovery of **OCx polynomial coefficients** used to estimate **chlorophyll-a concentration** from satellite reflectance data.

OCx algorithms estimate chlorophyll-a concentration (`chlor_a`) as a polynomial function of the logarithm of a ratio of **remote sensing reflectances (Rrs)** measured at different wavelengths:

$$
\log_{10}(\mathrm{chlor}_a)
=
a_0 + a_1 R + a_2 R^2 + a_3 R^3 + a_4 R^4
$$

where

$$
R = \log_{10}\left(\frac{R_{rs,\lambda_1}}{R_{rs,\lambda_2}}\right)
$$

Where $\lambda$ in this case corresponds to all the different wavelengths provided by the satellite (412, 443, 490, 510, 560, 665). 
and $a_0, \dots, a_4$ are **empirical polynomial coefficients**.

In this project, the OCx coefficients (usually derived through best-fit) are treated as **unknown parameters** and recovered by solving an **inverse optimization problem**.

- **Unknowns**: polynomial coefficients $\mathbf{a} = (a_0, \dots, a_4)$ 
- **Observations**: satellite-derived reflectance ratios and corresponding chlorophyll-a measurements
- **Forward model**: OCx polynomial evaluation
- **Objective**: minimize the discrepancy between predicted and observed chlorophyll-a values

Because of the fact that the *chlorophyll_a* values are themselves a product of the OCx algorithm, the starting error will be lower than usual in this experiment. 
This procedure should actually be done with data collected *in-situ*, so that one can better tune the OCx algorithm by using the parameters found. 

This experiment needs data from ESA datasets. You can retrieve it [here](https://mega.nz/file/HUkRwRjb#Fzpxo8oSJlPAZndPA8CH2r7K9HWKEagwD1johheyaNk) and place it in the data folder. The file should be named 'all_reflectances.npz' 

Standard PSO:
```
python3 inverse_ocx.py 
```
Fuzzy Self-Tuning PSO:
```
python3 inverse_ocx.py --model fuzzy 
```


