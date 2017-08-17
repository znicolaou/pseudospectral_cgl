# pseudospectral_cgl 
The file 2dcgleic.c is a C program that utilizes the GNU Scientific Library (GSL), the FFTW fast Fourier transform, 
and OpenMP to implement a parallel, pseudospectral integration of the two dimensional complex Ginzburg 
Landau equation. The file plot.nb is a Mathematica 11.1.1.0 notebook which will plot the output of 2dcgleic and can generate 
a grid refinement of a state for use as initial conditions. The file spiralic.dat is a sample initial condition file with
a grid size of 1536 used in the manuscript.

# System requirements
This code has been test with GSL 2.4 (https://www.gnu.org/software/gsl/) and FFTW 3.3.5 (http://www.fftw.org/) 
with the C compiler gcc 4.9.2 (https://gcc.gnu.org/).  

# Compiling and usage
Compile with

  gcc -fopenmp -O3 -o 2dcgleic 2dcgleic.c -lgsl -lgslcblas -lfftw3 -lfftw3_omp -lm
  
Once compiled, running the program will produce the following usage message:

usage: ./2dcgleic N L c1 c3 t1 t2 t3 dt epsabs epsrel Nthreads method out 

N is number of oscillators 
2Pi L is linear system size 
c1 is CGLE Laplacian coefficient 
c3 is CGLE cubic coefficient 
t1 is total integration time 
t2 is time to start averaging and outputting animation data 
t3 is time stop outputting animation data 
dt is the time between outputs 
epsabs is absolute error tolerance 
epsrel is relative error tolerance 
Nthreads number of threads to parallelize with
method is time stepping method - possible values are rk4 for explicit Runge-Kutta 4, rkf45 for Runge-Kutta-Fehlberg 4/5, and adams for Adams multistep
out is base file name for output and input.  The initial condition is retrieved from outic.dat if it exists; otherwise random initial conditions are used. The output files are: out.out contains time step data, outlast.dat contains the last state, outanimation.dat contains the states between timesteps between t2 and t3, outfrequency.dat contains the average frequency between t2 and t3 in dt steps, outvarfrequency.dat contains the variance of the frequency between t2 and t3 in dt steps, outorder.dat contains the average order parameter between t2 and t3 in dt steps, outvarorder.dat contains the variance of the order parameter between t2 and t3 in dt steps, amp contains the average amplitude between t2 and t3 in dt steps, outvaramp.dat contains the variance of the amplitude between t2 and t3 in dt steps. 

Example: ./2dcgleic 768 192 2.0 0.7 1e3 5e2 1e3 1 1e-3 1e-3 6 rkf45 random 
Example: ./2dcgleic 1536 192 2.0 0.85 1e4 1e3 1e4 1 1e-10 1e-10 6 rkf45 spiral
