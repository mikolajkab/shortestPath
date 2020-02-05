 #include <omp.h>
 #include <stdio.h>

 int main(int argc, char *argv[]) {

 int x, tid;
 x = 0;

 #pragma omp parallel shared(x) private(tid)
   {
   tid = omp_get_thread_num();

   #pragma omp critical 
   x = x + 1;
   printf("x = %d, thread id = %d\n", x, tid);

   }  /* end of parallel region */
   
   printf("x: ", x);

 }