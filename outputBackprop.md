

GitHub account :https://github.com/skabhilash10/mlblr_assignment2

To generate the random values for the matrix

```python
import numpy as np 

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
wh=np.random.rand(4,3)
```

```python
print(wh)
```

```python
bh=np.random.rand(1,3)
```

```python
print(bh)
```

```python
hidden_layer_input=(np.dot(x,wh))+bh
```

```python
print(hidden_layer_input)
```

```python
def sigmoid (hidden_layer_input): return 1/(1 + np.exp(-hidden_layer_input))  
```

```python
hiddenlayer_activations=sigmoid(hidden_layer_input)
```

```python
print(hiddenlayer_activations)
```

```python
wout=np.random.rand(3,1)
```

```python
print(wout)
```

```python
bout=np.random.rand(1,1)
```

```python
print(bout)
```

output_layer_input = matrix_dot_product (hiddenlayer_activations * wout ) + bout
output = sigmoid(output_layer_input)

```python
output_layer=np.dot(hiddenlayer_activations,wout)+bout
```

```python
output_layer_input=sigmoid(output_layer)
```

```python
print(output_layer_input)
```

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

```python
def sigmoid_(x): return x * (1 - x)             # derivative of sigmoid

```

```python
slope_output_layer=sigmoid_(E)
```

```python
print(slope_output_layer)
```

```python
slope_hidden_layer=sigmoid_(hiddenlayer_activations)
```

```python
print(slope_hidden_layer)
```

```python
d_output = E * slope_output_layer*0.1
```

```python
print(d_output)
```

```python
Error_at_hidden_layer=np.dot(d_output,np.transpose(wout))
```

```python
print(Error_at_hidden_layer)
```

```python
d_hiddenlayer= np.dot(Error_at_hidden_layer ,slope_hidden_layer)
print(d_hiddenlayer)

```

wout = wout + matrix_dot_product (hiddenlayer_activations.Transpose, d_output) *
learning_rate
wh = wh+ matrix_dot_product (X.Transpose,d_hiddenlayer) *
learning_rate

```python
wout_updated=wout + np.dot(np.transpose(hiddenlayer_activations), d_output) * 0.1
print(wout_updated)
```

```python
wh_updated=wh+np.dot(np.transpose(x),d_hiddenlayer)*0.1
print(wh_updated)
```

bh = bh + sum(d_hiddenlayer, axis=0) * learning_rate
bout = bout + sum(d_output,
axis=0)*learning_rate

```python
bh_updated= bh + (d_hiddenlayer) * 0.1
print(bh_updated)
```

```python
bout_updated = bout + (d_output)*0.1
print(bout_updated)
```
