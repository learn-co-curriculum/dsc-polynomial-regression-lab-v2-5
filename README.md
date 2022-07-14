# Polynomial Regression - Lab

## Introduction

In this lab, you'll practice your knowledge on adding polynomial terms to your regression model! 

## Objectives

You will be able to:

* Determine if polynomial regression would be useful for a specific model or set of data
* Create polynomial terms out of independent variables in linear regression

## Dataset

For this lab you'll be using some generated data:


```python
# Run this cell without changes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv('sample_data.csv')
df.head()
```


```python
# __SOLUTION__ 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv('sample_data.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.942870</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.121212</td>
      <td>-2.261629</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.242424</td>
      <td>3.100749</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.363636</td>
      <td>-0.285446</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.484848</td>
      <td>-1.012210</td>
    </tr>
  </tbody>
</table>
</div>



Let's check out a scatter plot of `x` vs. `y`: 


```python
# Run this cell without changes
df.plot.scatter(x="x", y="y");
```


```python
# __SOLUTION__
df.plot.scatter(x="x", y="y");
```


    
![png](index_files/index_11_0.png)
    


You will notice that the data is clearly of non-linear shape. Begin to think about what degree polynomial you believe will fit it best.

You will fit several different models with different polynomial degrees, then plot them in the same plot at the end.


```python
# Your code here - import StatsModels and separate the data into X and y

```


```python
# __SOLUTION__
import statsmodels.api as sm

X = df[["x"]]
y = df["y"]
```

## Build and Evaluate a Quadratic Model

This model should include a constant, `x`, and `x` squared. You can use `pandas` or `PolynomialFeatures` to create the squared term.


```python
# Your code here - prepare quadratic data and fit a model

```


```python
# __SOLUTION__
from sklearn.preprocessing import PolynomialFeatures

poly_2 = PolynomialFeatures(degree=2)

x_2 = pd.DataFrame(poly_2.fit_transform(X), columns=poly_2.get_feature_names(["x"]))
squared_results = sm.OLS(y, x_2).fit()
```


```python
# Your code here - evaluate (adjusted) R-Squared and coefficient p-values

```


```python
# __SOLUTION__
print(f"""
R-Squared Values
Quadratic Regression: {squared_results.rsquared_adj}
""")
```

    
    R-Squared Values
    Quadratic Regression: -0.015664188856822303
    



```python
# __SOLUTION__
squared_results.pvalues
```




    1      0.938055
    x      0.893974
    x^2    0.967420
    dtype: float64




```python
# Your written answer here - summarize findings

```

<details>
    <summary style="cursor: pointer"><b>Answer (click to reveal)</b></summary>
    
This is not a good model. Because we have multiple terms and are explaining so little of the variance in `y`, we actually have a negative adjusted R-Squared.

None of the coefficients are statistically significant at an alpha of 0.05
    
</details>

## Build and Evaluate a 4th Degree Polynomial Model

In other words, the model should include $x^0$ (intercept), $x^1$, $x^2$, $x^3$, and $x^4$ terms.

At this point we recommend importing and using `PolynomialFeatures` if you haven't already!


```python
# Your code here - prepare 4th degree polynomial data and fit a model

```


```python
# __SOLUTION__
poly_4 = PolynomialFeatures(degree=4)

x_4 = pd.DataFrame(poly_4.fit_transform(X), columns=poly_4.get_feature_names(["x"]))
poly_4_results = sm.OLS(y, x_4).fit()
```


```python
# Your code here - evaluate (adjusted) R-Squared and coefficient p-values

```


```python
# __SOLUTION__
print(f"""
R-Squared Values
Quadratic Regression:             {squared_results.rsquared_adj}
4th Degree Polynomial Regression: {poly_4_results.rsquared_adj}
""")
```

    
    R-Squared Values
    Quadratic Regression:             -0.015664188856822303
    4th Degree Polynomial Regression: 0.5667967820112239
    



```python
# __SOLUTION__
poly_4_results.pvalues
```




    1      1.995047e-04
    x      1.738988e-10
    x^2    3.340296e-14
    x^3    1.715785e-16
    x^4    7.408453e-18
    dtype: float64




```python
# Your written answer here - summarize findings

```

<details>
    <summary style="cursor: pointer"><b>Answer (click to reveal)</b></summary>
    
This is much better. We are explaining 57-58% of the variance in the target and all of our coefficients are statistically significant at an alpha of 0.05.
    
</details>

## Build and Evaluate an 8th Degree Polynomial Model

This model should include $x^0$ through $x^8$.


```python
# Your code here - prepare 8th degree polynomial data and fit a model

```


```python
# __SOLUTION__
poly_8 = PolynomialFeatures(degree=8)

x_8 = pd.DataFrame(poly_8.fit_transform(X), columns=poly_8.get_feature_names(["x"]))
poly_8_results = sm.OLS(y, x_8).fit()
```


```python
# Your code here - evaluate (adjusted) R-Squared and coefficient p-values

```


```python
# __SOLUTION__
print(f"""
R-Squared Values
Quadratic Regression:             {squared_results.rsquared_adj}
4th Degree Polynomial Regression: {poly_4_results.rsquared_adj}
8th Degree Polynomial Regression: {poly_8_results.rsquared_adj}
""")
```

    
    R-Squared Values
    Quadratic Regression:             -0.015664188856822303
    4th Degree Polynomial Regression: 0.5667967820112239
    8th Degree Polynomial Regression: 0.8188676291759689
    



```python
# __SOLUTION__
poly_8_results.pvalues
```




    1      0.683776
    x      0.618134
    x^2    0.371163
    x^3    0.199162
    x^4    0.144776
    x^5    0.157523
    x^6    0.225536
    x^7    0.357352
    x^8    0.554141
    dtype: float64




```python
# Your written answer here - summarize findings

```

<details>
    <summary style="cursor: pointer"><b>Answer (click to reveal)</b></summary>
    
Our R-Squared is higher, but none of the coefficients are statistically significant at an alpha of 0.05 any more. If what we care about is an inferential understanding of the data, this is too high a degree of the polynomial.
    
</details>

## Plot All Models

Build a single plot that shows the raw data as a scatter plot, as well as all of the models you have developed as line graphs. Make sure that everything is labeled so you can tell the different models apart!


```python
# Your code here

```


```python
# __SOLUTION__

fig, ax = plt.subplots(figsize=(10, 6))

models = [squared_results, poly_4_results, poly_8_results]
data = [x_2, x_4, x_8]
colors = ['yellow', 'lightgreen', 'cyan']

ax.scatter(X, y, label="data points", color="black")
for i, model in enumerate(models):
    ax.plot(
        X, # plot same x values for every model
        model.predict(data[i]), # generate predictions using relevant preprocessed data
        label=f"polynomial degree {(i + 1)*2}", # degree happens to be 2 times (i + 1)
        color=colors[i], # select color from list declared earlier
        linewidth=5,
        alpha=0.7
    )
    
ax.legend();
```


    
![png](index_files/index_41_0.png)
    


### Interpret Findings

Based on the metrics as well as the graphs, which model do you think is the best? Why?


```python
# Your written answer here

```

<details>
    <summary style="cursor: pointer"><b>Answer (click to reveal)</b></summary>
    
The quadratic model (polynomial degree 2) is definitely not the best based on all of the evidence we have. It has the worst R-Squared, the coefficient p-values are not significant, and you can see from the graph that there is a lot of variance in the data that it is not picking up on.

Our visual inspection aligns with the worse R-Squared for the 4th degree polynomial compared to the 8th degree polynomial. The 4th degree polynomial is flatter and doesn't seem to capture the extremes of the data as well.
    
However if we wanted to interpret the coefficients, then only the 4th degree polynomial has statistically significant results. The interpretation would be challenging because of the number of terms, but we could apply some calculus techniques to describe inflection points.

Overall it appears that this dataset is not particularly well suited to an inferential linear regression approach, even with polynomial transformations. So the "best" model could be either the 4th or 8th degree polynomial depending on which aspect of the model is more important to you, but either way it will be challenging to translate it into insights for stakeholders.
    
</details>

## Summary

Great job! You now know how to include polynomials in your linear models as well as the limitations of applying polynomial regression. 
