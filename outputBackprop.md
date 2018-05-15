To generate the random values for the matrix

```python
import numpy as np 
import pandas as pd 

```

```python
x=[[1,0,1,0],[1,0,1,1],[0,1,0,1]]
```

```python
x=np.array(x)
```

```python
print(x)
```

```python
print(pd.DataFrame(x))
```

Input X
-------------------
| 1 | 0| 1 | 0|
-------------------
|1  |0 | 1 | 1|
-------------------
|0 |1 | 0 | 1 |

```python
wh=np.random.rand(4,3)
```

```python
print(wh)
```

wh matrix

| 0.57187819 |0.49973999 |0.81024453|
-------------------------------------------
| 0.84671212 |0.08626766
|0.80960422|
-------------------------------------------
| 0.73248944 |0.3816215
|0.16557657|
-------------------------------------------
| 0.66565884 |0.5716475
|0.82226911|

```python
bh=np.random.rand(1,3)
```

```python
print(bh)
```

bh matrix 
| 0.74256293| 0.71502549| 0.91542495|

```python
hidden_layer_input=(np.dot(x,wh))+bh
```

```python
print(hidden_layer_input)
```

hidden_layer_input matrix 


| 2.04693056 |1.59638698 |1.89124605|
-------------------------------------------
| 2.71258941 |2.16803448
|2.71351516|
-------------------------------------------
| 2.25493389
|1.37294065 |2.54729828|

```python
def sigmoid (hidden_layer_input): return 1/(1 + np.exp(-hidden_layer_input))  
```

```python
hiddenlayer_activations=sigmoid(hidden_layer_input)
```

```python
print(hiddenlayer_activations)
```

hiddenlayer_activations matrix 


| 0.8856371 | 0.83151281 |0.86889754|
-------------------------------------------
| 0.93776544 |0.89734205
|0.93781945|
-------------------------------------------
| 0.90507527
|0.79785484 |0.9273918|

```python
wout=np.random.rand(3,1)
```

```python
print(wout)
```

wout matrix 
| 0.00396529 | 
----------------
| 0.99236788 |
----------------
|
0.23386983 |

```python
bout=np.random.rand(1,1)
```

```python
print(bout)
```

bout matrix 
|0.6605259|

```python
output_layer=np.dot(hiddenlayer_activations,wout)+bout
```

```python
output_layer_input=sigmoid(output_layer)
```

```python
print(output_layer_input)
```

output_layer_input matrix 
| 0.82188683 | 
----------------
| 0.8335274 |
----------------
| 0.81899297 |

```python
y=[[1],[1],[0]]
```

```python
y=np.array(y)
```

```python
E=y-output_layer_input
```

```python
print(E)
```

E matrix 
| 0.17811317 | 
----------------
| 0.1664726 |
----------------
|
-0.81899297 |

```python
def sigmoid_(x): return x * (1 - x)             # derivative of sigmoid

```

```python
slope_output_layer=sigmoid_(E)
```

```python
print(slope_output_layer)
```

slope_output_layer matrix 
| 0.14638887 | 
----------------
| 0.13875947 |
----------------
| -1.48974245 |

```python
slope_hidden_layer=sigmoid_(hiddenlayer_activations)
```

```python
print(slope_hidden_layer)
```

slope_hidden_layer matrix 


| 0.10128403 |0.14009926 |0.11391461|
-------------------------------------------
| 0.05836142 |0.0921193
|0.05831413|
-------------------------------------------
| 0.08591402
|0.16128249 |0.06733625|

```python
d_output = E * slope_output_layer*0.1
```

```python
print(d_output)
```

d_output matrix 
|0.00260738| 
----------------
| 0.00230996 |
----------------
|0.12200886 |

```python
Error_at_hidden_layer=np.dot(d_output,np.transpose(wout))
```

```python
print(Error_at_hidden_layer)
```

Error_at_hidden_layer matrix 


| 1.03390194e-05 |2.58747867e-03
|6.09787153e-04|
-------------------------------------------
| 9.15968757e-06
|2.29233502e-03 |5.40231100e-04|
-------------------------------------------
|
4.83800865e-04 |1.21077674e-01 |2.85341909e-02|

```python
d_hiddenlayer= np.dot(Error_at_hidden_layer ,slope_hidden_layer)
print(d_hiddenlayer)

```

d_hiddenlayer matrix 


| 0.00020445 |0.00033815 |0.00019313|
-------------------------------------------
| 0.00018113 |0.00029958 |0.0001711|
-------------------------------------------
| 0.00956675 |0.01582344
|0.00903704|

```python
wout_updated=wout + np.dot(np.transpose(hiddenlayer_activations), d_output) * 0.1
print(wout_updated)
```

wout_updated matrix 
|0.01545555| 
----------------
| 1.00252651 |
----------------
|0.24562802|

```python
wh_updated=wh+np.dot(np.transpose(x),d_hiddenlayer)*0.1
print(wh_updated)
```

wh_updated matrix 


| 0.57191675| 0.49980377| 0.81028095|
----------------------------------------
| 0.8476688|  0.08785 |   0.81050793|
-----------------------------------------
| 0.732528|   0.38168527| 0.16561299|
-----------------------------------------
| 0.66663363| 0.5732598 | 0.82318992|

```python
bh_updated= bh + (d_hiddenlayer) * 0.1
print(bh_updated)
```

bh_updated matrix 


| 0.74258337| 0.71505931| 0.91544426|
-------------------------------------------
| 0.74258104| 0.71505545|
0.91544206|
-------------------------------------------
| 0.7435196|
0.71660784| 0.91632865|

```python
bout_updated = bout + (d_output)*0.1
print(bout_updated)
```

bout_updated matrix 
|0.49755699| 
----------------
| 0.49755699 |
----------------
|0.50949714|
