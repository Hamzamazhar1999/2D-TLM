/*  -------------- Hamza Bin Mazhar - 20498276 ---------------
    -------------------- 2D-TLM using CUDA -------------------
    ----------- Coursework 02 - GPU Programming --------------*/

//Including all the libraries essentials
#include "cuda_runtime.h"   
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>  //for using output streaming to an external file
#include <iomanip>  // for setprecision
#include <time.h>   // for clock

#define M_PI 3.14276   //definition of pi value
#define c 299792458     //speed of light
#define mu0 M_PI*4e-7   //magnetic permeability
#define eta0 c*mu0      //impedance of the wave
#define square_root 1.41421356237 //square root of 2, constant defined to remove calculation again and again

#define NX 100          //first dimension of nodes (array X dimension)
#define NY 100          //second dimension of nodes (array Y dimension)
#define NT 8192          //number of time steps

/*Kernels for applying source, scattering, connecting and using probe to find the output*/
__global__ void tlmSource(double* dev_V1, double* dev_V2, double* dev_V3, double* dev_V4, int* dev_Ein, double E0);
__global__ void tlmScatter(double* dev_V1, double* dev_V2, double* dev_V3, double* dev_V4, double Z, int* dev_Ein, double E0);
__global__ void tlmConnect(double* V1, double* V2, double* V3, double* V4, double rXmin, double rXmax, double rYmin, double rYmax, int* dev_Eout);
__global__ void tlmOutput(double* dev_vout, double * dev_V2, double *dev_V4, int n, int* dev_Eout);
/*Function for declaring a 2d array using dynamic allocation*/
double** declare_2Darray(void);

using namespace std;

int main()
{
    clock_t start, end;     // defining clock variables for execution time calculation
    double cpu_time;

    // 2D mesh variables
    double I = 0, tempV = 0, E0 = 0, V = 0;
    double Z = eta0 / square_root;
    double dl = 1;      //seperation of nodes
    double dt = dl / (square_root * c);    //time step duration
    double width = 20 * dt * square_root;
    double delay = 100 * dt * square_root;

    /* Voltage Arrays for the host*/
    double** V1 = declare_2Darray();
    double** V2 = declare_2Darray();
    double** V3 = declare_2Darray();
    double** V4 = declare_2Darray();
    /* Voltage Arrays for the device*/
    double* dev_V1;
    double* dev_V2;
    double* dev_V3;
    double* dev_V4;
    /* Voltage Array dynamic memory allocation in GPU*/
    cudaMalloc((void**)&dev_V1, NX * NY * sizeof(double));
    cudaMalloc((void**)&dev_V2, NX * NY * sizeof(double));
    cudaMalloc((void**)&dev_V3, NX * NY * sizeof(double));
    cudaMalloc((void**)&dev_V4, NX * NY * sizeof(double));
    /* Voltage Arrays copied from host to device*/
    cudaMemcpy(dev_V1, V1[0], NX * NY * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_V2, V2[0], NX * NY * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_V3, V3[0], NX * NY * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_V4, V4[0], NX * NY * sizeof(double), cudaMemcpyHostToDevice);

    // boundary coefficients
    double rXmin = -1;
    double rXmax = -1;
    double rYmin = -1;
    double rYmax = -1;

    //Application of excitation voltage at this node
    int* Ein = (int*)malloc(2 * sizeof(int));
    Ein[0] = 10;
    Ein[1] = 10;
    //Monitoring Node is defined here
    int* Eout = (int*)malloc(2 * sizeof(int));
    Eout[0] = 15;
    Eout[1] = 15;
    /*defining the GPU data arrays to transfer these nodes to the GPU*/
    int* dev_Ein;
    int* dev_Eout;
    /* Simulation nodes dynamic memory allocation in GPU*/
    cudaMalloc((void**)&dev_Ein, 2 * sizeof(int));
    cudaMalloc((void**)&dev_Eout, 2 * sizeof(int));
    /* Simulation nodes copied from host to device*/
    cudaMemcpy(dev_Ein, Ein, 2 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Eout, Eout, 2 * sizeof(int), cudaMemcpyHostToDevice);

    /*Output Voltage Array of size NT*/
    double v_output[NT];
    for (int i = 0; i < NT; i++)
    {
        v_output[i] = 0;
    }
    /*Output Array defined on GPU*/
    double* dev_vout;
    /*Output Array GPU memory allocation to the size NT*/
    cudaMalloc((void**)&dev_vout, NT * sizeof(double));
    //Determining Kernel Size
    dim3 dimBlock(10, 10);
    dim3 dimGrid(ceil(NX / dimBlock.x), ceil(NY / dimBlock.y));
    //Creating an output file to log output voltages against the n*dt
    ofstream output("output_usingGPU.out");
    //Starting the TIME CALCULATION here for the actual loop for calculation
    start = clock();
    for (int n = 0; n < NT; n++) // Loop runs for the total time steps defined above
    {
        E0 = (1 / square_root) * exp(-(n * dt - delay) * (n * dt - delay) / (width * width));//Excitation Voltage 
        //Source Application Kernel
        tlmSource<<<1,1>>>(dev_V1, dev_V2,dev_V3, dev_V4, dev_Ein, E0);
        // Scattering Kernel
        tlmScatter << <dimGrid, dimBlock >> > (dev_V1, dev_V2, dev_V3, dev_V4, Z, dev_Ein,E0);
        // Connecting Kernel 
        tlmConnect << <dimGrid, dimBlock >> > (dev_V1, dev_V2, dev_V3, dev_V4, rXmin, rXmax, rYmin, rYmax,dev_Eout);
        // Output Probing Kernel 
        tlmOutput << <1, 1 >> > (dev_vout, dev_V2, dev_V4, n, dev_Eout);
    }
    /*Output Array from the Probing kernel is copied back to device for logging after loop ends*/
    cudaMemcpy(v_output, dev_vout, NT *  sizeof(double), cudaMemcpyDeviceToHost);
    //Printing outputs to an output file
    for (int n = 0; n < NT; n++) 
    {
        output << " n*dt" << "\t\t" << "Output" << endl;
        output << n * dt << "\t\t" << v_output[n] << endl;
    }
    end = clock();//ending the TIME CALCULATION here
    output.close();
    /*Freeing the defined variables and arrays*/
    cudaFree(dev_V1);
    cudaFree(dev_V2);
    cudaFree(dev_V3);
    cudaFree(dev_V4);
    cudaFree(dev_Ein);
    cudaFree(dev_Eout);
    cudaFree(dev_vout);

    /*Printing Time calculation for the main 2d algorithm here*/
    double TLM_Execution_Time = double(end - start) / double(CLOCKS_PER_SEC);
    cout << "Time taken by TLM algorithm using GPU is : " << fixed << TLM_Execution_Time << setprecision(5);
    cout << " sec " << endl;
    return 0;
}
//definition of the Source Kernel 
__global__ void tlmSource(double* dev_V1, double* dev_V2, double* dev_V3, double* dev_V4, int* dev_Ein, double E0)
{
    //Source Application is done with pointer-to-pointer ensuring indexing
    dev_V1[dev_Ein[0] * NX + dev_Ein[1]] = dev_V1[dev_Ein[0] * NX + dev_Ein[1]] + E0;
    dev_V2[dev_Ein[0] * NX + dev_Ein[1]] = dev_V2[dev_Ein[0] * NX + dev_Ein[1]] - E0;
    dev_V3[dev_Ein[0] * NX + dev_Ein[1]] = dev_V3[dev_Ein[0] * NX + dev_Ein[1]] - E0;
    dev_V4[dev_Ein[0] * NX + dev_Ein[1]] = dev_V4[dev_Ein[0] * NX + dev_Ein[1]] + E0;
}
//Definition of the scatter kernel
__global__ void tlmScatter(double* dev_V1, double* dev_V2, double* dev_V3, double* dev_V4, double Z,int* dev_Ein,double E0 )
{
    /*running on threads and blocks for most optimal application*/
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    /*Indexing over here is done to enssure pointer-to-pointer objects,
      so for an array dev_V1[x][y], the indexing is done like this: 
      dev_V1[x*NX+y].
     */
    if ((x*NX+y)<NX*NY) // breaking the CPU code into if statements for parallelization
    {
        double I = (2 * dev_V1[x * NX + y] + 2 * dev_V4[x * NX + y] - 2 * dev_V2[x * NX + y] - 2 * dev_V3[x * NX + y]) / (4 * Z);

        double V = 2 * dev_V1[x * NX + y] - I * Z;         // port1
        dev_V1[x * NX + y] = V - dev_V1[x * NX + y];
        V = 2 * dev_V2[x * NX + y] + I * Z;         // port2
        dev_V2[x * NX + y] = V - dev_V2[x * NX + y];
        V = 2 * dev_V3[x * NX + y] + I * Z;         // port3
        dev_V3[x * NX + y] = V - dev_V3[x * NX + y];
        V = 2 * dev_V4[x * NX + y] - I * Z;         // port4
        dev_V4[x * NX + y] = V - dev_V4[x * NX + y];
    }
}
//Definition of the connect kernel
__global__ void tlmConnect(double* dev_V1, double* dev_V2, double* dev_V3, double* dev_V4, double rXmin, double rXmax, double rYmin, double rYmax, int* dev_Eout)
{
    /*running on threads and blocks for most optimal application*/
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && (x * NX + y) < NX*NY)//Again, breaking the for loops from CPU code into if statements for parallelization
    {
        double tempV = dev_V2[x * NX + y];
        dev_V2[x * NX + y] = dev_V4[(x - 1) * NX + y];
        dev_V4[(x - 1) * NX + y] = tempV;
    }

    if (y > 0 && (x * NX + y) < NX * NY)
    {
        double tempV = dev_V1[x * NX + y];
        dev_V1[x * NX + y] = dev_V3[x * NX + (y - 1)];
        dev_V3[x * NX + (y - 1)] = tempV;
    }
    //Boundary Conditions applied here
    if (x < NX && y == NX - 1) // for maximum Y node dimension value
    {
        dev_V3[x * NX + (NY-1)] = rYmax * dev_V3[x * NX + (NY - 1)];
    }

    if (x < NX && y == 0) // for minimum Y node dimension value
    {
        dev_V1[x * NX + y] = rYmin * dev_V1[x*NX ];
    }

    if (x == NX - 1 && y < NX)// for maximum X node dimension value
    {
        dev_V4[(NX-1)*NX+y] = rXmax * dev_V4[(NX - 1) * NX + y];
    }

    if (x == 0 && y < NX)// for minimum X node dimension value
    {
        dev_V2[y] = rXmin * dev_V2[y];
    }
}
//Definition of Output Probing Kernel 
__global__ void tlmOutput(double* dev_vout, double* dev_V2, double* dev_V4, int n, int* dev_Eout)
{
    //probing the output as a sum of the dev_v2 and dev_V4 at 15,15 node using the Eout values
    dev_vout[n] = dev_V2[dev_Eout[0] + dev_Eout[1] * NX] + dev_V4[dev_Eout[0] + dev_Eout[1] * NX];
}
//2D dynamic allocation of the ararys 
double** declare_2Darray()
{
    double** V = new double* [NX];
    V[0] = new double[NX * NY];
    for (int i = 1; i < NX; ++i)
    {
        V[i] = V[i - 1] + NY;
    }
    for (int i = 0; i < NX; ++i) {
        for (int j = 0; j < NY; ++j) {
            V[i][j] = 0;
        }
    }
    return V;
}