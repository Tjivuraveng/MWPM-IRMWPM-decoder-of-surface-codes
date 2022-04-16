# A MWPM/IRMWPM Decoder for Surface Codes

Minimum weight perfect matching (MWPM) decoder is the most standard decoder of surface codes, but it contains some flaws when dealing with the noises in the depolarizing channels. Iteratively reweighted MWPM decoder is a modification to the conventional MWPM decoder and has better performence for depolarizing channels. For more information about IRMWPM decoder, please see 
https://doi.org/10.48550/arXiv.2202.11239.

## Download & Installation

Copy this repositoy and install opencv-python.

```console
pip3 install opencv-python
```

Create a new python file and import the module "SorCodSim".
```python
import SurCodSim as SCS
```
## Declare a Surface Code

- A 3x4 surface code with alternative boundaries

```python
L1=SCS.AB_Lattice(width=3,length=4)
```

- A 3x3 surface code with a hole (defect)
```python
hole = [(3,3)]                             # coordinate of the hole
L2 = SCS.D_Lattice(width=3,length=3,hole)
```
## Visualize the Lattices
```python
L1.show_lattice()
```
##### Output:
<img src=figures/fig1.PNG  width=32%>

```python
L2.show_lattice()
```
##### Output:
<img src=figures/fig2.PNG  width=24%>

## The Coordinate Systems 
<img src=figures/fig3.PNG  width=70%>

## Input Syndromes
```python
syndrome = [(1,0),(1,2),(3,0),(3,2),(0,1),(4,1)]
L1.Receiving_syndrome(syndrome)
L1.show_lattice()
```
##### Output:
<img src=figures/fig4.PNG  width=32%>

```python
syndrome = [(3,1),(5,1),(2,2),(4,0)]
L2.Receiving_syndrome(syndrome)
L2.show_lattice()
```
##### Output:
<img src=figures/fig5.PNG  width=24%>

## MWPM Decoding
```python
L1.MWPM_decoding()
```
##### Output:
```
X correction: 
 [[1 1]
 [3 1]]
Z correction: 
 [[2 0]
 [2 2]]
```
```python
L1.show_lattice()
```
##### Output:
<img src=figures/fig6.PNG  width=32%>

```python
L2.MWPM_decoding()
```

##### Output:
```
X correction: 
 [[4 1]]
Z correction: 
 [[2 1]
 [3 0]]
```

```python
L2.show_lattice()
```

##### Output:
<img src=figures/fig7.PNG  width=24%>

## IRMWPM Decoding

```python
L1.IRMWPM_decoding()
```

##### Output:

```
X correction: 
 [[1 1]
 [3 1]]
Z correction: 
 [[1 1]
 [3 1]]
```

```python
L1.show_lattice()
```

##### Output:
<img src=figures/fig8.PNG  width=32%>

```python
L2.IRMWPM_decoding()
```

##### Output:
```
X correction: 
 [[4 1]]
Z correction: 
 [[3 2]
 [4 1]]
```

```python
L2.show_lattice()
```
##### Output:
<img src=figures/fig9.PNG  width=24%>

## Reset the Lattices
```python
L1.clear_lattice()
L1.show_lattice()
```

##### Output:
<img src=figures/fig1.PNG  width=24%>

```python
L2.clear_lattice()
L2.show_lattice()
```

##### Output:
<img src=figures/fig2.PNG  width=24%>

