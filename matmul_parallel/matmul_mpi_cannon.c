#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

const int DEFAULT_M = 500, DEFAULT_N = 500, DEFAULT_K = 500;
const double E = 1e-7;
typedef int coord_t[2];

double drand(double low, double high)
{
    return ((double)rand() * (high - low)) / (double)RAND_MAX + low;
}

// calculate C = C + AB
// baseline algorithm
void matmul_ijk(const double *A, const double *B, double *C, int M, int N, int K) 
{
    for (int i = 0; i < M; i++) 
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < K; k++)
            {
                C[i * N + j] = C[i * N + j] + A[i * K + k] * B[k * N + j];
            }
        }
    }
}

void copy_send_buffer(const double *src, double *dest, const int m, const int n, const int b, const int lp, const int lq)
{
    for (int i = 0; i < b; i++)
    {
        for (int j = 0; j < b; j++)
        {
            dest[i * b + j] = src[(lp * b + i) * n + (lq * b + j)];
        }
    }
}

void copy_recv_buffer(const double *src, double *dest, const int m, const int n, const int b, const int lp, const int lq)
{
    for (int i = 0; i < b; i++)
    {
        for (int j = 0; j < b; j++)
        {
            dest[(lp * b + i) * n + (lq * b + j)] = src[i * b + j];
        }
    }
}

void distribute_AB(int isA, const double *src, double **dest, const int m, const int n, const int b, const coord_t coords, const int lrank, const int p, const int q, MPI_Comm comm_cart)
{
    *dest = (double *)malloc(b * b * sizeof(double));
    
    if (lrank == 0)
    {
        double **send_buffer = (double **)malloc(p * q * sizeof(double *));
        MPI_Request *request = (MPI_Request *)malloc(p * q * sizeof(MPI_Request));
        MPI_Status *status = (MPI_Status *)malloc(p * q * sizeof(MPI_Status));
        for (int i = 0; i < p; i++)
        {
            for (int j = 0; j < q; j++)
            {
                coord_t target_coords = {i, j};
                int target_rank;
                MPI_Cart_rank(comm_cart, target_coords, &target_rank);

                // get cannon offset
                int ii = 0, jj = 0;
                if (isA)
                    jj = i;
                else 
                    ii = j;

                if (target_rank == 0)
                {
                    copy_send_buffer(src, *dest, m, n, b, (i + ii) % p, (j + jj) % q);
                }
                else
                {
                    send_buffer[i * q + j] = (double *)malloc(b * b * sizeof(double));
                    copy_send_buffer(src, send_buffer[i * q + j], m, n, b, (i + ii) % p, (j + jj) % q);
                    MPI_Isend(send_buffer[i * q + j], b * b, MPI_DOUBLE, target_rank, 0, MPI_COMM_WORLD, &request[i * q + j]);
                }

            }
        }
        MPI_Waitall(p * q - 1, request + 1, status + 1);
        free(request);
        free(status);
        for (int i = 1; i < p * q; i++)
        {
            free(send_buffer[i]);
        }
        free(send_buffer);
    }
    else
    {
        MPI_Recv(*dest, b * b, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}


void distribute_AB_scatterv(int isA, const double *src, double **dest, const int m, const int n, const int b, const coord_t coords, const int lrank, const int p, const int q, MPI_Comm comm_cart)
{
    *dest = (double *)malloc(b * b * sizeof(double));
    
    if (lrank == 0)
    {
        MPI_Datatype tmp_type, block_type;
        MPI_Type_vector(b, b, n, MPI_DOUBLE, &tmp_type);
        MPI_Type_create_resized(tmp_type, 0, sizeof(double), &block_type);
        MPI_Type_commit(&block_type);
        int *sendcounts = (int *)malloc(p * q * sizeof(int));
        int *displs = (int *)malloc(p * q * sizeof(int));

        for (int i = 0; i < p * q; i++)
            sendcounts[i] = 1;
        for (int i = 0; i < p * q; i++)
        {
            coord_t target_coords;
            MPI_Cart_coords(comm_cart, i, 2, target_coords);
            int ii = target_coords[0], jj = target_coords[1];
            if (isA)
                jj = (jj + ii) % q;
            else 
                ii = (ii + jj) % p;
            displs[i] = b * ii * n + b * jj;
        }

        MPI_Scatterv(src, sendcounts, displs, block_type, *dest, b * b, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        free(sendcounts);
        free(displs);
    }
    else
    {
        MPI_Scatterv(NULL, NULL, NULL, NULL, *dest, b * b, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
}

void collect_C(const double *C, double *C0, int m, int n, int b, coord_t coords, int lrank, int p, int q, MPI_Comm comm_cart)
{
    if (lrank == 0)
    {
        double **recv_buffer = (double **)malloc(p * q * sizeof(double *));
        MPI_Request *request = (MPI_Request *)malloc(p * q * sizeof(MPI_Request));
        int *flag = (int *)calloc(p * q, sizeof(int));

        for (int send_rank = 1; send_rank < p * q; send_rank++)
        {
            recv_buffer[send_rank] = (double *)malloc(b * b * sizeof(double));
            MPI_Irecv(recv_buffer[send_rank], b * b, MPI_DOUBLE, send_rank, 0, MPI_COMM_WORLD, &request[send_rank]);
        }

        // copy rank0 C to C0
        copy_recv_buffer(C, C0, m, n, b, coords[0], coords[1]);

        // polling for other senders
        int cnt = 0;
        while (cnt < p * q - 1)
        {
            for (int send_rank = 1; send_rank < p * q; send_rank++)
            {
                if (flag[send_rank] == 0)
                {
                    MPI_Test(&request[send_rank], &flag[send_rank], MPI_STATUS_IGNORE);
                    if (flag[send_rank])
                    {
                        cnt++;
                        coord_t send_coords;
                        MPI_Cart_coords(comm_cart, send_rank, 2, send_coords);
                        copy_recv_buffer(recv_buffer[send_rank], C0, m, n, b, send_coords[0], send_coords[1]);
                    }
                }

            }
        }

        for (int i = 1; i < p * q; i++)
            free(recv_buffer[i]);
        free(recv_buffer);
        free(request);
        free(flag);
        
    }
    else
        MPI_Send(C, b * b, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
}

void collect_C_gatherv(const double *C, double *C0, int m, int n, int b, coord_t coords, int lrank, int p, int q, MPI_Comm comm_cart)
{
    MPI_Datatype tmp_type, block_type;
    MPI_Type_vector(b, b, n, MPI_DOUBLE, &tmp_type);
    MPI_Type_create_resized(tmp_type, 0, sizeof(double), &block_type);
    MPI_Type_commit(&block_type);

    int *recvcounts, *displs;
    if (lrank == 0)
    {
        recvcounts = malloc(p * q * sizeof(int));
        displs = malloc(p * q * sizeof(int));
        for (int i = 0; i < p * q; i++)
            recvcounts[i] = 1;
        for (int i = 0; i < p * q; i++)
        {
            coord_t target_coords;
            MPI_Cart_coords(comm_cart, i, 2, target_coords);
            int ii = target_coords[0], jj = target_coords[1];
            // there's no addtional displacement in C
            displs[i] = b * ii * n + b * jj;
        }    
    }
    MPI_Gatherv(C, b * b, MPI_DOUBLE, C0, recvcounts, displs, block_type, 0, MPI_COMM_WORLD);
    if (lrank == 0)
    {
        free(recvcounts);
        free(displs);
    }
}

int main(int argc, char **argv)
{
    srand(time(NULL));

    MPI_Init(&argc, &argv);
    
    int lrank, nrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &lrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nrank);

    // initial m, n, k
    int m, n, k;
    int p, q;
    switch (argc)
    {
        case 3: // main.c p q
            m = DEFAULT_M;
            n = DEFAULT_N;
            k = DEFAULT_K;
            p = atoi(argv[1]);
            q = atoi(argv[2]);
            break;
        case 6: // main.c p q m n k
            p = atoi(argv[1]);
            q = atoi(argv[2]);
            m = atoi(argv[3]);
            n = atoi(argv[4]);
            k = atoi(argv[5]);
            break;
        default:
            printf("Invaild input args.\n");
            return -1;
    }

    if (m != n || m != k || p != q) 
    {
        if (lrank == 0)
        {
            printf("Currently only support the case that M=N=K.\n");
        }
        return -1;
    }

    // initial virtual topology
    MPI_Comm comm_cart;
    {
        int dims[2] = {p, q};
        int periods[2] = {1, 1};
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm_cart);
    }
    coord_t coords;
    MPI_Cart_coords(comm_cart, lrank, 2, coords);
 
    // initial matrix and compute ground truth result
    double *A0, *B0, *C0, *D0, *A=NULL, *B=NULL, *C=NULL;
    if (lrank == 0)
    {
        A0 = (double *)malloc(m * k * sizeof(double));
        B0 = (double *)malloc(k * n * sizeof(double));
        C0 = (double *)calloc(m * n, sizeof(double));
        D0 = (double *)calloc(m * n, sizeof(double));
        for (int i = 0; i < m * k; i++)
        {
            A0[i] = drand(-1, 1);
        }
        for (int i = 0; i < k * n; i++)
        {
            B0[i] = drand(-1, 1);
        }
        matmul_ijk(A0, B0, D0, m, n, k);
    }

    int b = m / p;
    // first version: using send p2p commucation
    // distribute_AB(1, A0, &A, m, k, b, coords, lrank, p, q, comm_cart);
    // distribute_AB(0, B0, &B, k, n, b, coords, lrank, p, q, comm_cart);
    // second version: using scatterv
    distribute_AB_scatterv(1, A0, &A, m, k, b, coords, lrank, p, q, comm_cart);
    distribute_AB_scatterv(0, B0, &B, k, n, b, coords, lrank, p, q, comm_cart);
    C = (double *)calloc(b * b, sizeof(double));

    struct timeval begin, end;
    gettimeofday(&begin, 0);

    //kernel
    for (int iter = 0; iter < p - 1; iter++)
    {
        matmul_ijk(A, B, C, b, b, b);
        int rank_src, rank_dest;
        // shift A
        MPI_Cart_shift(comm_cart, 1, -1, &rank_src, &rank_dest);
         MPI_Sendrecv_replace(A, b * b, MPI_DOUBLE, rank_dest, 0, rank_src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // shift B
        MPI_Cart_shift(comm_cart, 0, 1, &rank_src, &rank_dest);
        MPI_Sendrecv_replace(B, b * b, MPI_DOUBLE, rank_dest, 0, rank_src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    matmul_ijk(A, B, C, b, b, b);
    
    // collect C to C0
    // first version: using point2point communication
    // collect_C(C, C0, m, n, b, coords, lrank, p, q, comm_cart);
    // second version: using gatherv
    collect_C_gatherv(C, C0, m, n, b, coords, lrank, p, q, comm_cart);

    // end kernel
    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&end, 0);

    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    double elapsed = seconds + microseconds * 1e-6;

    MPI_Finalize();

    if (lrank == 0)
    {
        // validation
        int flag = 0;
        for (int i = 0; i < m * n; i++)
        {
            if (fabs(C0[i] - D0[i]) > E)
            {
                printf("Validation failed.\n");
                flag = -1;
                break;
            }
        }
        if (flag == 0)
            printf("Validation succeeded.\n");
        printf("Elapsed %.6f seconds.\n", elapsed);
    }

    if (lrank == 0)
    {
        free(A0);
        free(B0);
        free(C0);
        free(D0);
    }
    free(A);
    free(B);
    free(C);


    return 0;
}