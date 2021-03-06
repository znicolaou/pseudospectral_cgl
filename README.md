# pseudospectral_cgl 
The file 2dcgleic.c is a C program that utilizes the GNU Scientific Library (GSL), the FFTW fast Fourier transform, 
and OpenMP to implement a parallel, pseudospectral integration of the two dimensional complex Ginzburg 
Landau equation. The file plot.nb is a Mathematica 11.1.1.0 notebook which will plot the output of 2dcgleic and can generate 
a grid refinement of a state for use as initial conditions. The file spiralic.dat (available in the release) is a sample
chimera initial condition file with a grid size of 1536.

# System requirements
This code has been run with GSL 2.4 (https://www.gnu.org/software/gsl/) and FFTW 3.3.5 (http://www.fftw.org/) 
with the C compiler gcc 4.9.2 (https://gcc.gnu.org/).  The terminal command gcc is actually the clang compiler in Mac, which does not support the -fopenmp option. For OpenMP functionality, Mac users should install gcc with homebrew, and use a command with a version number like gcc-7 to compile.

# Compiling and usage
Compile the program with

`gcc -fopenmp -O3 -o 2dcgleic 2dcgleic.c -lgsl -lgslcblas -lfftw3 -lfftw3_omp -lm`
  
Once compiled, running the program will produce a usage message:

`./2dcgleic`

usage: ./2dcgleic N L c1i c1f c3i c3f t1 t2 t3 dt epsabs epsrel Nthreads method out 

N is number of oscillators 

2Pi L is linear system size 

c1i is initial CGLE Laplacian coefficient 

c1f is final CGLE Laplacian coefficient

c3i is initial CGLE cubic coefficient 

c3f is final CGLE cubic coefficient 

t1 is total integration time 

t2 is time at which c1 and c3 reach their final values.  After this time, start averaging and outputting animation data. 

t3 is time stop outputting animation data 

dt is the time between outputs 

epsabs is absolute error tolerance 

epsrel is relative error tolerance 

Nthreads number of threads to parallelize with

method is time stepping method - possible values are rk4 for explicit Runge-Kutta 4, rkf45 for Runge-Kutta-Fehlberg 4/5, and adams for Adams multistep

out is base file name for output and input.  The initial condition is retrieved from outic.dat if it exists; otherwise random initial conditions are used. The output files are: out.out contains time step data, outlast.dat contains the last state, outanimation.dat contains the states between timesteps between t2 and t3, outfrequency.dat contains the average frequency between t2 and t3 in dt steps, outvarfrequency.dat contains the variance of the frequency between t2 and t3 in dt steps, outorder.dat contains the average order parameter between t2 and t3 in dt steps, outvarorder.dat contains the variance of the order parameter between t2 and t3 in dt steps, amp contains the average amplitude between t2 and t3 in dt steps, outvaramp.dat contains the variance of the amplitude between t2 and t3 in dt steps. 

-----------------------------------------------------------------------

Example 1) ./2dcgleic 768 192 2.0 2.0 0.72 0.72 1e3 5e2 5e2 1 1e-3 1e-3 4 rkf45 random 
A spiral is likely to nucleate out of amplitude turbulence with these parameters. If no spiral nucleation occurs or multiple spirals nucleate, try again. It may take a few attempts to get an isolated spiral. Use the Mathematica notebook plot.nb to create the spiralcoreic.dat initial condition, and then run the next example. 

-----------------------------------------------------------------------

Example 2) ./2dcgleic 768 192 2.0 2.0 0.72 0.85 1e3 5e2 5e2 1 1e-3 1e-3 4 rkf45 spiralcore 
Quasistatically increase c_3 to a value where the spiral nucleation rate is low.  The spiral should persist and no new spiral nucleation will occur once the parameters are changed. Use the Mathematica notebook plot.nb to center the spiral, refine the grid, create a new initial condition refineic.dat, and then run the next example. 

-----------------------------------------------------------------------
 
Example 3) ./2dcgleic 1536 192 2.0 2.0 0.85 0.85 5e3 4e3 5e3 1 1e-10 1e-10 4 rkf45 refine 
The grid spacing and error tolerances here are converged to the continuum limit, and after this run the spiral should have grown to its full size. WARNING: This will take over a day to run and will produce large (38GB) output files.  Use the Mathematica notebook plot.nb to create an animation of the frozen vortex chimera. 

-----------------------------------------------------------------------



