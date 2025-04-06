
# Wgpu Hartree-Fock


## My Progress So Far

```
SystemInfo { a1: [2.7652192, 1.5965], a2: [2.7652192, -1.5965], b1: [1.13611, 1.9678], b2: [1.13611, -1.9678], a0: 3.193, b0: 2.27222, delta: 0.0, t: -1.0, area_unit_cell: 0.0, _pad0: 0.0 }
# of k points: 262144
Trying chemical potential: 0
charge density 2.0
```

# Next ToDo
Draw energy band diagram for h0
(Just for the sake of checking if my shader programs are working correctly, where I implemented complex number/vector/matrix operations on my own.)

Fortunately I have `energy_eigen_buf` where this buffer stores 
- energy (2 bands)
- eigenstates
- k-vector

for all k-points (having k-vector here was for the sake of memory alignment; There was an implicit alignment padding which had the exact size of k-vector, so I just put it. Turned out it's gonna be useful when drawing graph)

## Thoughts

- I just realized I need to do a lot of linear algebra calculation with complex number, which I need to implement on my own; hard to generalize shader codes (e.g. dimensions of matrices)
    - And I even shouldn't generalize those things in shader code either
        - because conditional branching operations (if, loop, etc.) makes shader inefficient
- Learning a lot about wgpu through this.
    - Passing & casting data structures between Rust and WGSL
    - Learned about memory alignment rules
        - e.g. implicit alignment paddings
    - Buffer operatoins
        - copy data between (shader buffer) <-> (cpu buffer)
        - map & read from buffers
