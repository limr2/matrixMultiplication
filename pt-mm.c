/* Simple matrix multiply program
 *
 * Phil Nelson, March 5, 2019
 *
 */

#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>

/* idx macro calculates the correct 2-d based 1-d index
 * of a location (x,y) in an array that has col columns.
 * This is calculated in row major order.
 */

#define idx(x,y,col)  ((x)*(col) + (y))
int part;
int num_threads = 8;
pthread_barrier_t mybarrier;

/* Matrix storage */
double *A;
double *B;
double *C;
int x = 0, y = 0, z = 0;
int sTimes = 0;

/* Print a matrix: */
void MatPrint (double *A, int x, int y)
{
  int ix, iy;

  for (ix = 0; ix < x; ix++) {
    printf ("Row %d: ", ix);
    for (iy = 0; iy < y; iy++)
      printf (" %10.5G", A[idx(ix,iy,y)]);
    printf ("\n");
  }
}


/* Generate data for a matrix: */
void MatGen (double *A, int x, int y, int rand)
{
  int ix, iy;

  for (ix = 0; ix < x ; ix++) {
    for (iy = 0; iy < y ; iy++) {
      A[idx(ix,iy,y)] = ( rand ?
			  ((double)(random() % 200000000))/2000000000.0 :
			  (1.0 + (((double)ix)/100.0) + (((double)iy/1000.0))));
    }
  }
}

/* Print a help message on how to run the program */

void usage(char *prog)
{
  fprintf (stderr, "%s: [-dr] -x val [-y val] [-z val] [-n val] [-T]\n", prog);
  exit(1);
}

// START ROUTINE FOR MATMUL//
void *matmul(void *arg){
  long threadn = (long) arg;

  int sn = threadn * part;
  int en;

  // if last thread
  if(num_threads == threadn + 1){
    // do all that's left
    en = (x * z) - sn;
  }
  else{
    en = part;
  }

  div_t output = div(sn, x);
  int ix = output.quot;
  int jx = output.rem;

  for (; ix < x; ix++) {
    // Rows of solution
    for (; jx < z; jx++) {
      // Columns of solution
      float tval = 0;
      for (int kx = 0; kx < y; kx++) {
	       // Sum the A row time B column
	       tval += A[idx(ix,kx,y)] * B[idx(kx,jx,z)];
      }
      C[idx(ix,jx,z)] = tval;
      en--;
      if(en == 0){
        pthread_exit(0);
      }
    }
    jx = 0;
  }
  pthread_exit(0);
}

// START ROUTINE FOR MATSQUARE //
void *matsquare(void *arg){
  long threadn = (long) arg;

  // first thread decrements sTimes
  if(threadn == 0){
    sTimes--;
  }

  int sn = threadn * part;
  int en;

  // if last thread
  if(num_threads == threadn + 1){
    // do rest of work
    en = (x * x) - sn;
  }
  else{
    en = part;
  }

  div_t output = div(sn, x);
  int ix = output.quot;
  int jx = output.rem;

  for (; ix < x; ix++) {
    // Rows of solution
    for (; jx < x; jx++) {
      // Columns of solution
      float tval = 0;
      for (int kx = 0; kx < x; kx++) {
	       // Sum the A row time B column
	       tval += A[idx(ix,kx,x)] * A[idx(kx,jx,x)];
      }
      B[idx(ix,jx,x)] = tval;
      en--;
      // if done
      if(en == 0){
        pthread_barrier_wait(&mybarrier);
        if(sTimes > 0){
          matsquare(arg);
        }
        else{
          pthread_exit(0);
        }
      }
    }
    jx = 0;
  }
  pthread_exit(0);
}

void MatMul ()
{
  long i;
  int err;
  pthread_t ids[num_threads];

  // minimum one result per thread
  if(num_threads > (x * z)){
    num_threads = x * z;
  }

  // number of results per thread
  part = (x * z) / num_threads;

  for (i = 0; i < num_threads; i++) {
    err = pthread_create (&ids[i], NULL, matmul, (void *)i);
    if (err) {
      fprintf (stderr, "Can't create thread %ld\n", i);
      exit (1);
    }
  }

  for (i = 0; i < num_threads; i++) {
  pthread_join(ids[i], NULL);
  }

  return;
}

void MatSquare ()
{
  long i;
  int err;
  pthread_t ids[num_threads];

  pthread_barrier_init(&mybarrier, NULL, num_threads);

  // minimum one result per thread
  if(num_threads > (x * x)){
    num_threads = x * x;
  }

  // number of results per thread
  part = (x * x) / num_threads;

  for (i = 0; i < num_threads; i++) {
    err = pthread_create (&ids[i], NULL, matsquare, (void *)i);
    if (err) {
      fprintf (stderr, "Can't create thread %ld\n", i);
      exit (1);
    }
  }

  for (i = 0; i < num_threads; i++) {
  pthread_join(ids[i], NULL);
  }

  return;
}

/* Main function
 *
 *  args:  -d   -- debug and print results
 *         -r   -- use random data between 0 and 1
 *         -s t -- square the matrix t times
 *         -x   -- rows of the first matrix, r & c for squaring
 *         -y   -- cols of A, rows of B
 *         -z   -- cols of B
 *         -n   -- use n threads
 *         -T   -- time computation
 *
 */

int main (int argc, char ** argv)
{
  extern char *optarg;   /* defined by getopt(3) */
  int ch;                /* for use with getopt(3) */

  /* option data */
  int debug = 0;
  int square = 0;
  int useRand = 0;
  int timecomp = 0;

  clock_t start1;
  clock_t end1;
  double cpu_time;
  int clock_time;
  struct timeval start, end;
  struct timeval result;

  while ((ch = getopt(argc, argv, "drs:x:y:z:n:T")) != -1) {
    switch (ch) {
    case 'd':  /* debug */
      debug = 1;
      break;
    case 'r':  /* debug */
      useRand = 1;
      srandom(time(NULL));
      break;
    case 's':  /* s times */
      sTimes = atoi(optarg);
      square = 1;
      break;
    case 'x':  /* x size */
      x = atoi(optarg);
      break;
    case 'y':  /* y size */
      y = atoi(optarg);
      break;
    case 'z':  /* z size */
      z = atoi(optarg);
      break;
    case 'n': /* n threads */
      num_threads = atoi(optarg);
      break;
    case 'T':
      timecomp = 1;
      break;
    case '?': /* help */
    default:
      usage(argv[0]);
    }
  }

  /* verify options are correct. */
  if (square) {
    if (y != 0 || z != 0 || x <= 0 || sTimes < 1) {
      fprintf (stderr, "Inconsistent options\n");
      usage(argv[0]);
    }
  } else if (x <= 0 || y <= 0 || z <= 0) {
    fprintf (stderr, "x, y, and z all need to be specified.\n");
    usage(argv[0]);
  }

  // MatSquare
  if (square) {
    A = (double *) malloc (sizeof(double) * x * x);
    B = (double *) malloc (sizeof(double) * x * x);
    MatGen(A,x,x,useRand);

    if (timecomp){
      start1 = clock();
      gettimeofday(&start, NULL);
    }

    MatSquare();

    if (timecomp){
      end1 = clock();
      gettimeofday(&end, NULL);
    }

    // print them
    if (debug) {
      printf ("-------------- orignal matrix ------------------\n");
      MatPrint(A,x,x);
      printf ("--------------  result matrix ------------------\n");
      MatPrint(B,x,x);
    }
    pthread_barrier_destroy(&mybarrier);
  // MatMul
  } else {

    A = (double *) malloc (sizeof(double) * x * y);
    B = (double *) malloc (sizeof(double) * y * z);
    C = (double *) malloc (sizeof(double) * x * z);
    MatGen(A,x,y,useRand);
    MatGen(B,y,z,useRand);

    if (timecomp){
      start1 = clock();
      gettimeofday(&start, NULL);
    }

    MatMul();

    if (timecomp){
      end1 = clock();
      gettimeofday(&end, NULL);
    }

    if (debug) {
      printf ("-------------- orignal A matrix ------------------\n");
      MatPrint(A,x,y);
      printf ("-------------- orignal B matrix ------------------\n");
      MatPrint(B,y,z);
      printf ("--------------  result C matrix ------------------\n");
      MatPrint(C,x,z);
    }
  }
  if (timecomp){
    cpu_time = ((double) (end1 - start1)) / CLOCKS_PER_SEC;
    timersub(&start, &end, &result);
    clock_time = result.tv_sec;
    printf("\n  cpu time: %f\n", cpu_time);
    printf("clock time: %d\n", clock_time);
  }
  return 0;
}
