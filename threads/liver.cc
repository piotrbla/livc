#include <math.h>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define min(lhs,rhs)    ((lhs) < (rhs) ? (lhs) : (rhs))
#define max(lhs,rhs)    ((lhs) > (rhs) ? (lhs) : (rhs))
#define floord(val,d) (((val)<0) ? -((-(val)+(d)-1)/(d)) : (val)/(d))
#define ceild(val,d)  ceil(((double)(val))/((double)(d)))


bool is_files_equal(const char * , const char * ) ;
void print_files_equal(const char * , const char * );
void write_results_full(int, double, char);
void write_results(int, double );
void write_results_last(int, double );
int** allocate_matrix(int) ;

int **get_full_copy(int ** , int);
int* allocate_vector(int);
int *get_vector_copy(int *, int);

void deallocate_matrix(int **, int ) ;
void print_vector(int* , int );
void print_matrix(int**, int);

void LMKernel6_04(int loop, int n, int *input_w, int **input_b)
{
  int** b = get_full_copy(input_b, n);
  int* w = get_vector_copy(input_w, n);

  double start = omp_get_wtime();

for (int counter = 1; counter <= loop; counter += 1)
  for (int var1 = 1; var1 < n; var1 += 1)
    for (int var2 = var1; var2 < n; var2 += 1)
       w[var2] += b[-var1 + var2][var2] * w[(var2 - (-var1 + var2)) - 1];


  double execution_time = omp_get_wtime() - start;

  printf("MOD_04: %lf\n", execution_time);
  write_results(n, execution_time);
  print_vector(w, n);
  deallocate_matrix(b, n);
  free(w);
  return;
}

void LMKernel6_04_Modification_DAPT(int loop, int n, int *input_w, int **input_b)
{
  int** b = get_full_copy(input_b, n);
  int* w = get_vector_copy(input_w, n);

  double start = omp_get_wtime();
  
for (int i0 = 1; i0 <= loop; i0 += 1) {
  for (int w0 = 0; w0 <= floord(n - 1, 27); w0 += 1) {
    #pragma omp parallel for num_threads(MYTHREADS)
    for (int h0 = max(0, w0 - (n + 53) / 54 + 1); h0 <= w0 / 2; h0 += 1) {
      for (int i1 = max(1, 54 * h0); i1 <= min(n - 1, 54 * h0 + 53); i1 += 1) {
        for (int i2 = max(54 * w0 - 54 * h0, i1); i2 <= min(n - 1, 54 * w0 - 54 * h0 + 53); i2 += 1) {
          w[i2] += (b[-i1 + i2][i2] * w[i1 - 1]);
        }
      }
    }
  }
}

  double execution_time = omp_get_wtime() - start;

  printf("MOD_05: %lf\n", execution_time);
  write_results_last(n, execution_time);
  print_vector(w, n);
  deallocate_matrix(b, n);
  free(w);
  return;
}

#define PERFORMANCE_TEST 1

void make_work_one_size(const int ZMAX, const int LMAX)
{
  int** input_b = allocate_matrix(ZMAX);
  int* result_w = allocate_vector(ZMAX);
  for (int i = 0; i < ZMAX; i++)
    for (int j = 0; j < ZMAX; j++)
      input_b[i][j] = 0;
  for (int i = 0; i < ZMAX; i++)
    result_w[i] = 1;
  const char* seqTest = "1234432432123412";
#if PERFORMANCE_TEST==1
  for (int i = 0; i < ZMAX; i++)
    for (int j = 0; j < ZMAX; j++)
      input_b[i][j] = rand()%4+1;
#else
  for (int i = 0; i < ZMAX; i++)
    for (int j = 0; j < ZMAX; j++)
      input_b[i][j] = seqTest[i]-'0';
      
#endif
  LMKernel6_04(LMAX, ZMAX, result_w, input_b);
  LMKernel6_04_Modification_DAPT(LMAX, ZMAX, result_w, input_b);
  deallocate_matrix(input_b, ZMAX);
  free(result_w);
}


int main()
{
#if PERFORMANCE_TEST==1
    const int ZMAX = 5500;
    const int LMAX = 100;
    for (int z = 5000 ; z <ZMAX ; z+=500)
      make_work_one_size(z, LMAX);
#else
  const int ZMAX = 16;
  const int LMAX = 10;
  make_work_one_size(ZMAX, LMAX);
#endif 
  
}

bool is_files_equal(const char * filename_template, const char * filename_compared) {
  std::ifstream f1(filename_template, std::ifstream::binary | std::ifstream::ate);
  std::ifstream f2(filename_compared, std::ifstream::binary | std::ifstream::ate);
  if (f1.fail() || f2.fail()) {
    return false;
  }
  if (f1.tellg() != f2.tellg()) {
    return false;
  }
  f1.seekg(0, std::ifstream::beg);
  f2.seekg(0, std::ifstream::beg);
  return std::equal(std::istreambuf_iterator <char >(f1.rdbuf()),
    std::istreambuf_iterator <char >(),
    std::istreambuf_iterator <char >(f2.rdbuf()));
}

void print_files_equal(const char* filename_template, const char* filename_compared)
{
  if (is_files_equal(filename_template, filename_compared))
    std::cout << filename_compared << ": OK\n";
  else
    std::cout << filename_compared << ": ERROR\n";
}

void write_results_full(int n, double execution_time, char end_char)
{
  FILE* f = fopen("results.txt", "at");
  fprintf(f, "%d:%lf%c", n, execution_time, end_char);
  fclose(f);
}

void write_results(int n, double execution_time)
{
  write_results_full(n, execution_time, ';');
}

void write_results_last(int n, double execution_time)
{
  write_results_full(n, execution_time, '\n');
}

int** allocate_matrix(int N) {
  int** t = (int**)malloc(sizeof(int*) * N);
  for (int i = 0; i < N; i++) {
    t[i] = (int*)malloc(sizeof(int) * N);
  }
  return t;
}

int **get_full_copy(int ** table, int N)
{
  int **S = allocate_matrix(N);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      S[i][j] = table[i][j];
  return S;
}

int* allocate_vector(int N) {
  int* t = (int*)malloc(sizeof(int) * N);
  return t;
}

int *get_vector_copy(int *table, int N)
{
  int *S = allocate_vector(N);
  for (int i = 0; i < N; i++)
      S[i] = table[i];
  return S;
}


void deallocate_matrix(int **t, int N) {
  for (int i = 0; i < N; i++) {
    free(t[i]);
  }
  free(t);
}

void print_vector(int* vector, int N) {
  static int fileno=1;
  char filename[12];
  sprintf(filename, "resVec_%d", fileno);
  FILE* f = fopen(filename, "wt");
  for (int i = 0; i < N; i++) {
    fprintf(f, "%d ", vector[i]);
    fprintf(f, "\n");
  }
  fclose(f);
  fileno++;
}

void print_matrix(int** matrix, int N) {
  static int fileno=1;
  char filename[10];
  sprintf(filename, "resMat_%d", fileno);
  FILE* f = fopen(filename, "wt");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++)
      fprintf(f, "%d ", matrix[i][j]);
    fprintf(f, "\n");
  }
  fclose(f);
  fileno++;
}

