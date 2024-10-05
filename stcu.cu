#include <iostream>
#include <cuda.h>
#include <string>
#include <cublas_v2.h>
#include <cassert>
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include <bitset>
using namespace std;

/*********************************** definitions **********************************/
#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) 
{
   if (stat != CUBLAS_STATUS_SUCCESS) 
   {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}
#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}
// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) 
{
   if (stat != cudaSuccess) 
   {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

/************************************* kernels *************************************/
__global__ void mma_sp_16x8x32_general(float* d, __half *a_vals, __half  *b, float  *c, uint32_t* e, int m, int k, int n)
{
    //// A is row-major B is col-major

    //// each block of 32 (8x4) threads is handling a 16x8 tile of the result matrix
    int CY_nblocks = n/8; // Each result block has a width of 8 

    int CBlkXIdx = blockIdx.x / CY_nblocks;
    int CBlkYIdx = blockIdx.x % CY_nblocks;
    int CBlkWidth = 8;
    int CBlkHeight = 16;

    int ABlkWidth = 16; // width of A block
    int ABlkHeight = 16; // height of A block

    int BBlkWidth = 32; // width of B transposed
    int BBlkHeight = 8; // height of B transposed
    
    int tid = threadIdx.x;
    int outer = tid / 4; // m or n dimension
    int inner = tid % 4; // k dimension
    
    //// locating the indices of C fragment
    int cd_idx0 = (CBlkXIdx * CBlkHeight + outer) * n + CBlkYIdx * CBlkWidth + 2 * inner;
    int cd_idx1 = cd_idx0 + 1;
    int cd_idx2 = (CBlkXIdx * CBlkHeight + outer + 8) * n + CBlkYIdx * CBlkWidth + 2 * inner;
    int cd_idx3 = cd_idx2 + 1;

    float C0 = c[cd_idx0], C1 = c[cd_idx1], C2 = c[cd_idx2], C3 = c[cd_idx3];
    float D0 = 0, D1 = 0, D2 = 0, D3 = 0;
    for(int innerBlkIdx = 0; innerBlkIdx < (k/32); innerBlkIdx++)
    {
        //// Compute the row and col of a thread of a warp given a 8x4 thread block
        //// Compute linear offsets into each matrix
        int a_idx0 = (CBlkXIdx * ABlkHeight + outer) * (k/2) + innerBlkIdx * ABlkWidth + 2 * inner; //k/2 as A is compressed format
        int a_idx1 = a_idx0 + 1;
        int a_idx2 = a_idx0 + 8;
        int a_idx3 = a_idx0 + 9;
        int a_idx4 = (CBlkXIdx * ABlkHeight + outer + 8) * (k/2) + innerBlkIdx * ABlkWidth + 2 * inner;
        int a_idx5 = a_idx4 + 1;
        int a_idx6 = a_idx4 + 8;
        int a_idx7 = a_idx4 + 9;
        
        int b_idx0 = (CBlkYIdx * BBlkHeight  + outer) * k + innerBlkIdx * BBlkWidth + 2 * inner;
        int b_idx1 = b_idx0 + 1;
        int b_idx2 = b_idx0 + 8;
        int b_idx3 = b_idx0 + 9;
        int b_idx4 = b_idx0 + 16;
        int b_idx5 = b_idx0 + 17;
        int b_idx6 = b_idx0 + 24;
        int b_idx7 = b_idx0 + 25;
        
        __half aArray[8]={a_vals[a_idx0], a_vals[a_idx1], a_vals[a_idx4], a_vals[a_idx5], a_vals[a_idx2], a_vals[a_idx3], a_vals[a_idx6], a_vals[a_idx7]};    

        //// based on the inctruction documentation, 2 threads 0 and 1 are carrying the meta data for the threads 0,1,2,3
        //// thread 0 is carrying the meta data for the first 16 cols of the rows 0 of the A matrix (at t0.E.low16Bits) and the first 16 cols of the row 8 of the A matrix (at t0.E.high16Bits)
        //// thread 1 is carrying the meta data for the last  16 cols of the rows 0 of the A matrix (at t1.E.low16Bits) and the first 16 cols of the row 8 of the A matrix (at t1.E.high16Bits)
        uint32_t E = 0;

        uint32_t E1 = e[(CBlkXIdx * CBlkHeight + outer) * (k/32) + innerBlkIdx];        // e.g. E[0]
        uint32_t E2 = e[(CBlkXIdx * CBlkHeight + outer + 8) * (k/32) + innerBlkIdx];    // e.g. E[8]
        if(tid % 4 == 0) // tid = 0,4,8,12,...
        {
            E = (E1 & 0x0000FFFF) | ((E2 << 16) & 0xFFFF0000); // E = concat (E2.low, E1.low)
        }
        else if(tid % 4 == 1) // tid = 1,5,9,13,...
        {
            E = (E1 >> 16) | (E2 & 0xFFFF0000); // E = concat (E2.high, E1.high)
        }
        
        __half bArray[8]={b[b_idx0], b[b_idx1], b[b_idx2], b[b_idx3], b[b_idx4], b[b_idx5], b[b_idx6], b[b_idx7]};
        
        uint32_t const *A = reinterpret_cast<uint32_t const *>(&aArray);
        uint32_t const *B = reinterpret_cast<uint32_t const *>(&bArray);
        float d0 = 0, d1 = 0, d2 = 0, d3 = 0; 

        //// Issue Tensor Core operation
        asm volatile(
            "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
            "{%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;\n"
            : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
                "r"(B[2]), "r"(B[3]), "f"(C0), "f"(C1), "f"(C2), "f"(C3),
                "r"(E));
        D0 += d0;
        D1 += d1;
        D2 += d2;
        D3 += d3;
        //// printf("\n====> kernel DEBUG info for blockIdx.x:%d tid: %d, E:0x%x\n\n",blockIdx.x ,threadIdx.x, E);
    }
    d[cd_idx0] += D0;
    d[cd_idx1] += D1;
    d[cd_idx2] += D2;
    d[cd_idx3] += D3;
}
/************************************* helpers *************************************/
template <class T>
void random_dense_init(T* mat, int nrows, int ncols, uint32_t max_value, int Pattern=0, T fillVal= (T)0)
{
    // generates a matrix of uniformly random values in range [0,max_value]
    // cout<<"==================================\n";
    // cout<<"Generating random dense matrix...\n";
    if(Pattern == 0) // random fill with max_value
    {
        srand(time(0));
        // srand(0);
        for(int i = 0; i < nrows*ncols; i++)
        { 
            mat[i] = rand() % max_value;
        }
    }
    else if(Pattern == 1) // fixed fill with fillVal
    {
        for(int i = 0; i < nrows*ncols; i++)
        { 
            mat[i] = (T)fillVal;
        }
    }
    else if(Pattern == 2) // 1,2,3,4, ...
    {
        for(int i = 0; i < nrows*ncols; i++)
        { 
            mat[i] = (T)i;
        }
    }
    else if(Pattern == 3) // 1, 1, 1, ...;2, 2, 2, ...;... 
    {
        for(int i = 0; i < nrows; i++)
        { 
            for(int j = 0 ; j < ncols; j++)
            {
                mat[i * ncols + j] = (T)i;
            }
        }
    }
    else if(Pattern == 4) // 1, 1, 1, ...;2, 2, 2, ...;... 
    {
        for(int i = 0; i < nrows; i+=(nrows/2))
        { 
            int r = 2*i/nrows; // 0,1
            for(int j = 0 ; j < ncols; j+=(ncols/2))
            {
                int c = 2*j/ncols; //0,1
                for(int ii = 0; ii<nrows/2; ii++)
                    for(int jj = 0; jj<ncols/2; jj++)
                    {
                        int rr = i+ii;
                        int cc = j + jj;
                        mat[rr * ncols + cc] = (T)(2*r+c+1);
                    }
            }
        }
    }
    else if(Pattern == 5)
    {
        for(int i = 0; i < nrows; i++)
        { 
            for(int j = 0 ; j < ncols; j++)
            {
                mat[i * ncols + j] = (T)i;
            }
        }
    } 
}
void generate_rnd_metadata(uint32_t* col_idxs, int m, int n)
{
    // generates mxn random 2 bit colIdxs packed as uint32_t 
    // every 16 element is packed as a uint32_t
    // srand(0);
    srand(time(0));
    // uint32_t test[3] = {0xddddeeee, 0xdedededd,0xeeeeeeee };
    uint32_t valid_hex_digits[6] = {0x4,0x8,0xc,0x9,0xd,0xe}; // valid sequences = {01 ~> 0100 = 0x4,02 ~>1000=0x8,03 ~> 1100=0xc,12 ~> 1001 = 0x9,13 ~> 1101=0xd, 23 ~> 1101=0xe}
    for(int i = 0; i<(m*n/16); i++)
    {
        uint32_t randHexNum = 0;
        for(int d  = 0; d<8; d++) // each uint_32 is 8 hex digits
        {
            randHexNum = (randHexNum<<4)| valid_hex_digits[rand() % 6];
        }
        // col_idxs[i] = test[1];
        col_idxs[i] = randHexNum;
    }
}

template <class T>
void print_dense(T* mat, int nrows, int ncols, int format)
{
    cout<<"printing the dense Matrix "<<nrows<<"x"<<ncols<<"\n\n";
    for(int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            if(format == 0)
                cout<<"0b"<<bitset<8*sizeof(__half)>(mat[ncols * i + j])<<", ";
            else if(format == 1)
                printf("0x%x, ", (int)mat[ncols * i + j]);
            else if(format == 2)
                printf("%d, ", mat[ncols * i + j]);
        }
        cout<<endl;
    }
    cout<<"=================================="<<endl;
}
template <>
void print_dense(__half* mat, int nrows, int ncols, int format)
{
    cout<<"printing the dense Matrix "<<nrows<<"x"<<ncols<<"\n\n";
    for(int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            if(format == 0)
                cout<<"0b"<<bitset<8*sizeof(__half)>(mat[ncols * i + j])<<", ";
            else if(format == 1)
                printf("0x%x, ",(uint32_t)mat[ncols * i + j]);
            else if(format == 2)
            {
                printf("%.f, ",(float) mat[ncols * i + j]);
            }
        }
        cout<<endl;
    }
    cout<<"=================================="<<endl;
}
template <>
void print_dense(float* mat, int nrows, int ncols, int format)
{
    cout<<"printing the dense Matrix "<<nrows<<"x"<<ncols<<"\n\n";
    for(int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            if(format == 0)
                cout<<"0b"<<bitset<8*sizeof(__half)>(mat[ncols * i + j])<<", ";
            else if(format == 1)
                printf("0x%x, ",(int)mat[ncols * i + j]);
            else if(format == 2)
                printf("%.2f, ",(float) mat[ncols * i + j]);
        }
        cout<<endl;
    }
    cout<<"=================================="<<endl;
}

void NextPerm(int* A, int j)
{
    int t = A[j];
    A[j] = A[j+1];
    A[j+1] = t;
}
bool match(float* C, float* ans, int m , int n)
{
    float epsilon = 1e-6;
    for(int i = 0; i< m; i++)
    {
        for(int j = 0; j<n; j++)
        {
            if(abs(ans[i*n + j] - C[i*n +j]) > epsilon)
            {
                // cout<<"test failed at: i="<<i<<" j="<<j<<endl;
                // cout<<"ans:"<<ans[i*n + j]<<" != C value:"<<C[i*n + j]<<" diff:"<< (ans[i*n + j] - C[i*n + j])<<endl;
                return false;
            }
        }
    }
    return true;
}

template <class T>
void fillSparse(T* dense, uint32_t* col_idxs, T* sparse , int m, int n)
{
    cout<<"fillSparse method\n";
    for(int i = 0 ; i< m; i++)
    {
        for(int j = 0; j<n; j+=2)
        {
            uint32_t E = col_idxs[i*(n/16) + (j/16)];
            // uint32_t group4Bits = E>> 4*(8-j/2-1) &(0xF);
            uint32_t group4Bits = E>> 4*(j/2) &(0xF);
            uint32_t low2Bits = group4Bits & (0x3);
            uint32_t hi2Bits = group4Bits>>2;
            uint32_t LowOffset = i*(2*n) + (2*j+low2Bits);
            uint32_t HiOffset  = i*(2*n) + (2*j+hi2Bits);
            // printf("i:%d, j:%d, E[%d][%d]:0x%x, group4Bits:0x%x, hi2bits:%x, low2bits:%x, LowOffset:%d, HiOffset:%d\n", i, j, i, j/16, E, group4Bits, hi2Bits, low2Bits, LowOffset, HiOffset);
            sparse[LowOffset] = dense [i*n+j];
            sparse[HiOffset]  = dense [i*n + j+1];
        }
        // cout<<endl;
    }

}

void serialMatMul(__half* A, __half*BT, float* C, int M, int N, int K)
{
    for(int i = 0; i< M; i++)
    {
        for (int j = 0; j< N; j++)
        {
            float sum = 0;
            for( int k = 0; k < K; k++)
            {
                sum+= (float)A[i*K+k] * (float)BT[j*K+k];
            }
            C[i*N+j] = sum;
        }
    }
}
/*********************************  host functions *********************************/
float GPU_Compute_Cublas(uint32_t* hA, uint32_t* hB, int32_t* hC , int m, int k, int n,int iters)
{
        
    // half *a_fp16;
    // half *b_fp16;
    // float *c_cublas;
    // float *c_host_cublas;
    float alpha = 1.0f;
    float beta  = 0.0f;
    cublasHandle_t cublasHandle;
    cublasErrCheck(cublasCreate(&cublasHandle));

    //--------------------------------------------------------------------------
    // Device memory management
    auto     A_size         = m * k * sizeof(uint32_t);
    auto     B_size         = k * n * sizeof(uint32_t);
    auto     C_size         = m * n * sizeof(int32_t);
    auto     lda            = k;//(is_rowmajor) ? num_A_cols : num_A_rows;
    auto     ldb            = n;//(is_rowmajor) ? num_B_cols : num_B_rows;
    auto     ldc            = n;//(is_rowmajor) ? num_C_cols : num_C_rows;
    uint32_t *dA, *dB;
    int32_t *dC;
    CHECK_CUDA( cudaMalloc((void**) &dA, A_size) )
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size) )
    CHECK_CUDA( cudaMalloc((void**) &dC, C_size) )

    CHECK_CUDA( cudaMemcpy(dA, hA, A_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size, cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // printf("Running with cuBLAS...\n");
    float elapsed_time = 0, elapsed_time_partial = 0 ;
    
    for(int iter = 0; iter< iters;iter++)
    {
        cudaEvent_t startcublas, stopcublas;
        cudaErrCheck(cudaEventCreate(&startcublas));
        cudaErrCheck(cudaEventCreate(&stopcublas));
        cudaErrCheck(cudaEventRecord(startcublas));
        cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
            m,n,k,
            &alpha,
            dA, CUDA_R_32I, lda,
            dB, CUDA_R_32I, ldb,
            &beta,
            dC, CUDA_R_64U, ldc,
            CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));
        cudaErrCheck(cudaEventRecord(stopcublas));
        cudaErrCheck(cudaEventSynchronize(stopcublas));
        cudaErrCheck(cudaEventElapsedTime(&elapsed_time_partial, startcublas, stopcublas))
        elapsed_time += elapsed_time_partial;
    }
    // copy back the results
    CHECK_CUDA( cudaMemcpy(hC, dC, C_size, cudaMemcpyDeviceToHost) )    
    
    CHECK_CUDA( cudaFree(dA) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    cublasErrCheck(cublasDestroy(cublasHandle))
    return  elapsed_time/iters;
}

float GPU_mma_sp_16x8x32_f16_general(__half* hA_vals, uint32_t* hA_colIdxs, __half* hB, float* hC, int m, int k, int n,int iters)
{
    __half * d_A;
    uint32_t * d_AE;
    __half* d_B;
    float* d_C;
    float* d_D;

    //===========  allocate device memory
    cudaError_t err = cudaSuccess;

    cudaMalloc((void**) &d_A, ((m*k/2)  * sizeof( __half )));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate d_A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //// 2 bits/element is needed to be able to locate its location in uncompressed 2:4 sparse matrix
    //// m*(k/2) * 2bits/ 32bits = m*k/32 uint32_t
    cudaMalloc((void**) &d_AE, ((m*k/32)  * sizeof( uint32_t )));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate d_AE (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaMalloc((void**) &d_B, ((k*n)  * sizeof( __half )));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate d_B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaMalloc((void**) &d_C, ((m*n)  * sizeof( float )));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate d_C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaMalloc((void**) &d_D, ((m*n)  * sizeof( float )));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate d_D (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //===========  copy data to the device
    err = cudaMemcpy(d_A, hA_vals, ((m*k/2)  * sizeof( __half )), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy hA_vals from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMemcpy(d_AE, hA_colIdxs, ((m*k/32)  * sizeof( uint32_t )), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy hA_colIdxs from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, hB, ((k*n)  * sizeof( __half )), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cout<<"Lunching mma_sp_16x8x32_general kernel ...\n";
    //=========== lunch the kernel
    int numberOfBlocks = (m*n)/(16*8);
    int threadsPerBlock = 32;
    cout<<"numberOfBlocks: "<<numberOfBlocks<<endl;
    cout<<"threadsPerBlock: "<<threadsPerBlock<<endl;
    float elapsed_time = 0, elapsed_time_partial = 0 ;
    for(int i = 0; i < iters; i++)
    {
        cudaMemset(d_D, 0, (m*n)*sizeof(float));
        cudaEvent_t start, stop;
        cudaErrCheck(cudaEventCreate(&start));
        cudaErrCheck(cudaEventCreate(&stop));
        cudaErrCheck(cudaEventRecord(start));
        mma_sp_16x8x32_general<<<numberOfBlocks, threadsPerBlock>>>(d_D, d_A, d_B,d_C,d_AE, m, k, n);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch mma kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        cudaErrCheck(cudaEventRecord(stop));
        cudaErrCheck(cudaEventSynchronize(stop));
        cudaErrCheck(cudaEventElapsedTime(&elapsed_time_partial, start, stop))
        elapsed_time += elapsed_time_partial;
    }
    //=========== copy back the results 
    err = cudaMemcpy(hC, d_D, ((m*n)  * sizeof( float )), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy A from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    // free device memory
    CHECK_CUDA(cudaFree(d_A))
    CHECK_CUDA(cudaFree(d_B))
    CHECK_CUDA(cudaFree(d_C))
    CHECK_CUDA(cudaFree(d_D))

    return elapsed_time/iters;
}
/**************************************  main **************************************/
int main(int argc, char** argv)
{
    ////// ################################## handle the input parameters #################################
    if(argc < 2 || argc > 4)
    {
        cout<<"Usage Error: ./STCU.o <iters> [test=false] [DEBUG=false]\n";
        cout<<"e.g.: stcu 10 1\n";
        cout<<"e.g.: stcu 10\n";
        assert(false && "Usage Error: ./STCU.o <iters> [test=false] [DEBUG=false]");
    }
    int iters = atoi(argv[1]);
    assert(iters > 0 && "iters must be greater than 0");
    bool test = false;
    bool DEBUG = false;
    if(argc >= 3)
    {
        test = (bool) atoi(argv[2]);
        cout<<"test mode enabled.\n";
        if(argc == 4)
        {
            DEBUG = (bool) atoi(argv[3]);
            cout<<"DEBUG mode enabled.\n";
        }
    }


    ////// ################################ input matrices dimensions #####################################
    int M, N, K;
    if(DEBUG)
    {
        // M = 16;
        // N = 8;
        // K = 32;

        // M = 16;
        // N = 8;
        // K = 64;

        // M = 16;
        // N = 8;
        // K = 128;

        M = 128;
        N = 128;
        K = 128;
    }
    else
    {
        M = 1024*8;
        N = 1024;
        K = 1024*128;
    }

    cout<<"\n\n################  Dims: ################\n";
    cout<<"M:"<<M<<"\tN:"<<N<<"\tK:"<<K<<endl;

    ////// ############################## allocate host copies of HA, HB, HC ##############################
    __half *HA_sp_vals_f16  = new __half[M*K/2]; // the compressed A_vals
    //// 2 bits/element is needed to be able to locate its location in uncompressed 2:4 sparse matrix
    //// M*(K/2) * 2bits/ 32bits = M*K/32 uint32_t
    uint32_t *HA_sp_colIdxs_packed_u32  = new uint32_t[M*K/32];
    __half *HBT_sp_f16 = new __half[K*N];
    float* HC_sp_f16   = new float[M*N];

    ////// ################################# Random fill HA and HB  #######################################
    //// generate A_vals M X ceil(k/2)
    // random_dense_init(HA_sp_vals_f16,  M, (K+2-1)/2, 10,1, __half(1));      // fill 1
    // random_dense_init(HA_sp_vals_f16,  M, (K+2-1)/2, 10,3);              // fill with row number
    // random_dense_init(HA_sp_vals_f16,  M, (K+2-1)/2, 10,2);                 // fill sequentially
    random_dense_init(HA_sp_vals_f16,  M, (K+2-1)/2, 100,0);              // fill randomly

    generate_rnd_metadata(HA_sp_colIdxs_packed_u32, M, (K+2-1)/2);

    //// generate BT
    // random_dense_init(HBT_sp_f16, N, K, 10,1, __half(1));   // fill 1 ==> ok
    // random_dense_init(HBT_sp_f16, N, K, 10,3);              // fill with row number ===>ok
    // random_dense_init(HBT_sp_f16, N, K, 10,2);              // fill sequentially
    random_dense_init(HBT_sp_f16, N, K, 100,0);              // fill random
    // random_dense_init(HBT_sp_f16, N, K, 10,5);              // 
    // uint32_t* HBT_sp_f16_u32packed  = reinterpret_cast<uint32_t  *>(HBT_sp_f16);

    if(DEBUG)
    {
        cout<<"\n\n################  HA_sp_vals_f16 HA_sp_vals_f16 matrix ################\n\n";
        print_dense(HA_sp_vals_f16, M, (K+2-1)/2, 2);
        // print_dense(HA_sp_vals_f16, M, (K+2-1)/2, 1);
        // print_dense(HA_sp_vals_f16_u32packed, M, (K+2-1)/4, 1);

        cout<<"\n\n################  HA_sp_colIdxs_packed_u32 A u32packed matrix ################\n\n";
        print_dense(HA_sp_colIdxs_packed_u32, M, (K+32-1)/32, 1);

        cout<<"\n\n################  HBT_sp_f16 BT matrix ################\n\n";
        print_dense(HBT_sp_f16, N, K, 2);
        // print_dense(HBT_sp_f16, N, K, 1);
        // print_dense(HBT_sp_f16_u32packed, N, (K+2-1)/2, 1);
    }

    ////// ################################ call the kernel luncher  #######################################
    float time = 0;
    // time = GPU_mma_sp_16x8x32_f16(HA_sp_vals_f16, HA_sp_colIdxs_packed_u32, HBT_sp_f16, HC_sp_f16, M, K, N, iters);
    time = GPU_mma_sp_16x8x32_f16_general(HA_sp_vals_f16, HA_sp_colIdxs_packed_u32, HBT_sp_f16, HC_sp_f16, M, K, N, iters);
    cout<<"time (sec):"<<time<<endl;

    if(test)
    {
        cout<<"testing ...\n";
        __half * sparseA = new __half[M*K]();
        fillSparse(HA_sp_vals_f16,HA_sp_colIdxs_packed_u32, sparseA, M, K/2);
        if(DEBUG)
        {
            cout<<M<<"sparseA\n";
            print_dense(sparseA, M, K, 2);
        }
        float* C_serial = new float[M*N];
        serialMatMul(sparseA, HBT_sp_f16, C_serial, M, N, K);
        if(match(C_serial, HC_sp_f16, M, N))
        {
            cout<<"test passed!\n";
        }
        else
        {
            cout<<"test failed!\n";
            if(DEBUG)
            {
                cout<<"\n================ CPU results:\n";
                print_dense(C_serial, M, N, 2);
                cout<<"\n================STCU results:\n";
                print_dense(HC_sp_f16, M, N, 2);
            }
        }
    }

    return 0;
}
