# 2D-TLM
Porting a 2D Transmission-Line Matrix Algorithm to CUDA

TLM or Transmission Line matrix method is a numerical technique that employs the time domain
to provide an approximation of the electromagnetic wave propagation. This approach uses a
cartesian matrix of nodes to depict the two-dimensional space where wave propagation and
scattering take place. It revolves around the discretization of the propagation of electromagnetic
waves in both time and space. This is an iterative procedure with the two steps of scattering and
connecting as its key components. TLM offers a highly instructive technique to use this algorithm
in computer simulations and wave propagation modeling.

The 2D CPU code provided as a reference for port to GPU has portions that could be very well
parallelized and optimized for runtime speedups and optimized application. The first thing that is
noticeable is the definition of compute extensive and repeating applications like the sqrt() function.

The most important parts that can be parallelized are the scatter and connect functions. In a
Transmission Line Matrix (TLM) simulation, the scatter function is used to update the grid of
nodes at each time step. Based on the neighbors’ values and the TLM equation's coefficients, it
calculates the new values for the nodes.

The grid has the dimensions 'Nx x Ny'. Each node in the grid has four different port voltages: 'V1',
'V2', 'V3', and 'V4'. The inner loop computes the updated values for the port voltages at each node
while the outer loop iterates across the rows and columns of the grid. The port voltages
and impedance 'Z' are used in calculation of the current 'I' flowing through the node. The current
and impedances are then used to update the port voltages.
The next part of the code that can be parallelized is the connect function. The structure of the
simulated network is set using this function. It describes the connections between the nodes of the
grids. The coefficients of the TLM equation are set up using the connect functions and these
ultimately lead to the update of the node values.

# Changes made in the GPU kernel

Initially, the connect function is used to exchange the V2 and V4 port voltage values between the
grid nodes. Then the exchange of the voltages at the V1 and V3 port happens. These transactions
establish the connection of the grid nodes.

The nodes along the grid's edges are subject to boundary conditions according to the boundary
function. It multiplies the V3 and V1 port voltages for the nodes at the top and bottom borders of
the grid by the corresponding boundary reflection coefficients, rYmax and rYmin. It multiplies the
V4 and V2 port voltages for the nodes at the left and right edges of the grid by the corresponding
boundary reflection coefficients, rXmax and rXmin.

Other than these two functions, the source and the output probing functions could also be done
using CUDA kernels to ensure a seamless data allocation in device without having to copy and
allocate new memory locations after the updates.

Furthermore, there are some inconsistencies in the code that could be dealt with by using simple
coding techniques to properly optimize memory allocation and memory access. These include
dynamic allocation of Ein[] and Eout[] arrays to allow easy memory copying from host to device
for further processing and evaluating the output

# Improvements from GPU Use:
![image](https://github.com/Hamzamazhar1999/2D-TLM/assets/129704102/75fcacc2-c3d9-41b9-aeb6-3f412f903fd9)


