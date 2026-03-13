// assignment.cu — CUDA Streams & Events: Bluesky firehose word frequency
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>

#define NUM_BINS 65536
#define NUM_STREAMS_DEFAULT 4
#define REPEAT 5000  // work amplification per word char

// ============================================================================
// GPU Kernels
// ============================================================================

__device__ int is_word_char(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '\'';
}

__device__ unsigned int fnv1a_lower(const unsigned char *data,
                                    int start, int len) {
    unsigned int h = 2166136261u;
    for (int j = 0; j < len; j++) {
        unsigned char c = data[start + j];
        if (c >= 'A' && c <= 'Z') c += 32;
        h ^= c;
        h *= 16777619u;
    }
    return h;
}

// Word frequency — hash words into histogram via global atomics
__global__ void word_frequency(const unsigned char *data,
                               int n,
                               unsigned int *histogram,
                               unsigned int *hash_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride) {
        // Only fire at word-start positions
        int cur = is_word_char(data[i]);
        int prev = (i > 0) ? is_word_char(data[i - 1]) : 0;
        if (!cur || prev) continue;

        // Scan to word end
        int end = i;
        while (end < n && is_word_char(data[end])) end++;
        int wlen = end - i;
        if (wlen < 2 || wlen > 40) continue;

        // Hash and count
        unsigned int h = fnv1a_lower(data, i, wlen);
        atomicAdd(&histogram[h % NUM_BINS], 1);

        // Work amplification: FNV-1a rounds per character
        unsigned int work = 2166136261u;
        for (int c = 0; c < wlen; c++) {
            for (int r = 0; r < REPEAT; r++) {
                work ^= data[i + c];
                work *= 16777619u;
            }
        }
        if (idx == 0) { *hash_out = work; }
    }
}

// ============================================================================
// CPU Functions
// ============================================================================

// Read all bytes from a FILE* (works for both files and stdin)
unsigned char *read_all(FILE *fp, int *out_size) {
    size_t capacity = 1 << 20;  // 1 MB initial
    size_t size = 0;
    unsigned char *buf = (unsigned char *)malloc(capacity);
    if (!buf) { fprintf(stderr, "malloc failed\n"); exit(1); }

    while (1) {
        size_t n = fread(buf + size, 1, capacity - size, fp);
        size += n;
        if (n == 0) break;
        if (size == capacity) {
            capacity *= 2;
            buf = (unsigned char *)realloc(buf, capacity);
            if (!buf) { fprintf(stderr, "realloc failed\n"); exit(1); }
        }
    }
    *out_size = (int)size;
    return buf;
}

struct StreamResult {
    float h2d_ms;
    float kernel_ms;
    float d2h_ms;
    float total_ms;
};
struct PipelineResult {
    StreamResult *streams;
    int num_streams;
    float total_ms;
    unsigned int histogram[NUM_BINS];
    int data_bytes;
};

// Run the CUDA streams pipeline with event timing
void run_pipeline(const unsigned char *h_data, int n,
                  int num_streams, int blockSize, int totalThreads,
                  PipelineResult *result) {
    int numBlocks = (totalThreads + blockSize - 1) / blockSize;
    int chunk_size = (n + num_streams - 1) / num_streams;
    size_t hist_bytes = NUM_BINS * sizeof(unsigned int);

    result->num_streams = num_streams;
    result->data_bytes = n;
    result->streams = (StreamResult *)malloc(
        num_streams * sizeof(StreamResult));
    memset(result->histogram, 0, hist_bytes);

    // Per-stream host (pinned) and device allocations
    unsigned char **h_chunks = (unsigned char **)malloc(
        num_streams * sizeof(unsigned char *));
    unsigned int **h_hists = (unsigned int **)malloc(
        num_streams * sizeof(unsigned int *));
    unsigned char **d_chunks = (unsigned char **)malloc(
        num_streams * sizeof(unsigned char *));
    unsigned int **d_hists = (unsigned int **)malloc(
        num_streams * sizeof(unsigned int *));
    unsigned int **d_hash = (unsigned int **)malloc(
        num_streams * sizeof(unsigned int *));
    cudaStream_t *streams = (cudaStream_t *)malloc(
        num_streams * sizeof(cudaStream_t));

    // Events: start, after-h2d, after-kernel, after-d2h per stream
    cudaEvent_t *e_start = (cudaEvent_t *)malloc(
        num_streams * sizeof(cudaEvent_t));
    cudaEvent_t *e_h2d = (cudaEvent_t *)malloc(
        num_streams * sizeof(cudaEvent_t));
    cudaEvent_t *e_kern = (cudaEvent_t *)malloc(
        num_streams * sizeof(cudaEvent_t));
    cudaEvent_t *e_d2h = (cudaEvent_t *)malloc(
        num_streams * sizeof(cudaEvent_t));
    cudaEvent_t e_total_start, e_total_end;
    cudaEventCreate(&e_total_start);
    cudaEventCreate(&e_total_end);

    for (int s = 0; s < num_streams; s++) {
        int lo = s * chunk_size;
        int hi = lo + chunk_size;
        if (hi > n) hi = n;
        int sz = hi - lo;

        cudaHostAlloc(
            (void **)&h_chunks[s], sz, cudaHostAllocDefault);
        memcpy(h_chunks[s], h_data + lo, sz);

        cudaHostAlloc(
            (void **)&h_hists[s], hist_bytes, cudaHostAllocDefault);
        memset(h_hists[s], 0, hist_bytes);

        cudaMalloc(&d_chunks[s], sz);
        cudaMalloc(&d_hists[s], hist_bytes);
        cudaMemset(d_hists[s], 0, hist_bytes);
        cudaMalloc(&d_hash[s], sizeof(unsigned int));
        cudaMemset(d_hash[s], 0, sizeof(unsigned int));

        cudaStreamCreate(&streams[s]);
        cudaEventCreate(&e_start[s]);
        cudaEventCreate(&e_h2d[s]);
        cudaEventCreate(&e_kern[s]);
        cudaEventCreate(&e_d2h[s]);
    }

    // Launch pipeline
    cudaEventRecord(e_total_start);

    for (int s = 0; s < num_streams; s++) {
        int lo = s * chunk_size;
        int hi = lo + chunk_size;
        if (hi > n) hi = n;
        int sz = hi - lo;
        int grid = (sz + blockSize - 1) / blockSize;
        if (grid > numBlocks) grid = numBlocks;
        if (grid < 1) grid = 1;

        cudaEventRecord(e_start[s], streams[s]);
        cudaMemcpyAsync(
            d_chunks[s], h_chunks[s], sz,
            cudaMemcpyHostToDevice, streams[s]);
        cudaEventRecord(e_h2d[s], streams[s]);

        word_frequency<<<grid, blockSize, 0, streams[s]>>>(
            d_chunks[s], sz, d_hists[s], d_hash[s]);
        cudaEventRecord(e_kern[s], streams[s]);

        cudaMemcpyAsync(
            h_hists[s], d_hists[s], hist_bytes,
            cudaMemcpyDeviceToHost, streams[s]);
        cudaEventRecord(e_d2h[s], streams[s]);
    }

    cudaEventRecord(e_total_end);
    cudaEventSynchronize(e_total_end);

    // Collect per-stream timing
    for (int s = 0; s < num_streams; s++) {
        cudaEventElapsedTime(
            &result->streams[s].h2d_ms, e_start[s], e_h2d[s]);
        cudaEventElapsedTime(
            &result->streams[s].kernel_ms, e_h2d[s], e_kern[s]);
        cudaEventElapsedTime(
            &result->streams[s].d2h_ms, e_kern[s], e_d2h[s]);
        cudaEventElapsedTime(
            &result->streams[s].total_ms, e_start[s], e_d2h[s]);
    }
    cudaEventElapsedTime(
        &result->total_ms, e_total_start, e_total_end);

    // Merge histograms
    for (int s = 0; s < num_streams; s++) {
        for (int b = 0; b < NUM_BINS; b++) {
            result->histogram[b] += h_hists[s][b];
        }
    }

    // Cleanup
    for (int s = 0; s < num_streams; s++) {
        cudaFreeHost(h_chunks[s]);
        cudaFreeHost(h_hists[s]);
        cudaFree(d_chunks[s]);
        cudaFree(d_hists[s]);
        cudaFree(d_hash[s]);
        cudaStreamDestroy(streams[s]);
        cudaEventDestroy(e_start[s]);
        cudaEventDestroy(e_h2d[s]);
        cudaEventDestroy(e_kern[s]);
        cudaEventDestroy(e_d2h[s]);
    }
    cudaEventDestroy(e_total_start);
    cudaEventDestroy(e_total_end);

    free(h_chunks); free(h_hists);
    free(d_chunks); free(d_hists); free(d_hash);
    free(streams);
    free(e_start); free(e_h2d); free(e_kern); free(e_d2h);
}

// Print per-stream timing table
void print_stream_table(const PipelineResult *r) {
    printf("  %-8s %8s %8s %8s %8s  (ms)\n",
           "Stream", "H2D", "Kernel", "D2H", "Total");
    for (int s = 0; s < r->num_streams; s++) {
        printf("  %-8d %8.3f %8.3f %8.3f %8.3f\n", s,
               r->streams[s].h2d_ms, r->streams[s].kernel_ms,
               r->streams[s].d2h_ms, r->streams[s].total_ms);
    }
    printf("  Total: %.3f ms  (%.1f MB/s)\n",
           r->total_ms, (float)r->data_bytes / r->total_ms / 1e3f);
}

// Print top-k histogram bins by count
void print_top_bins(const unsigned int *histogram, int top_k) {
    // Copy to scratch so we can zero out found bins
    unsigned int tmp[NUM_BINS];
    memcpy(tmp, histogram, sizeof(tmp));
    printf("\n  Top %d histogram bins:\n", top_k);
    for (int t = 0; t < top_k; t++) {
        int best = -1;
        unsigned int best_val = 0;
        for (int b = 0; b < NUM_BINS; b++) {
            if (tmp[b] > best_val) { best = b; best_val = tmp[b]; }
        }
        if (best < 0) break;
        printf("    bin %5d : %6u\n", best, best_val);
        tmp[best] = 0;
    }
}

// ============================================================================
// Main
// ============================================================================

void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s <threads> <block_size> --file PATH "
        "[--streams N] [--csv PATH]\n", prog);
}

// Print run configuration and device info
void print_config(int totalThreads, int blockSize, int numBlocks,
                  int numStreams, const char *filePath) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Bluesky Firehose Histogram - CUDA Streams & Events\n");
    printf("  device: %s, threads: %d, blocks: %dx%d, streams: %d\n",
           prop.name, totalThreads, numBlocks, blockSize, numStreams);
    printf("  input: %s\n\n", filePath);
}

// Print sequential vs concurrent comparison
void print_comparison(const PipelineResult *r_seq,
                      const PipelineResult *r_conc) {
    float speedup = r_seq->total_ms / r_conc->total_ms;
    int match = 1;
    for (int i = 0; i < NUM_BINS; i++) {
        if (r_seq->histogram[i] != r_conc->histogram[i]) { match = 0; break; }
    }

    printf("\n  Comparison\n");
    printf("  Sequential : %10.3f ms\n", r_seq->total_ms);
    printf("  Concurrent : %10.3f ms\n", r_conc->total_ms);
    printf("  Speedup    : %10.2fx\n", speedup);
    printf("  Histograms : %s\n", match ? "match" : "MISMATCH");

    print_top_bins(r_conc->histogram, 10);
}

// Append summary row to CSV file
void write_csv_row(const char *csvPath, int totalThreads, int blockSize,
                   int numStreams, const PipelineResult *r_seq,
                   const PipelineResult *r_conc) {
    FILE *csv = fopen(csvPath, "a");
    if (!csv) return;
    float speedup = r_seq->total_ms / r_conc->total_ms;
    fprintf(csv, "%d,%d,%d,%.4f,%.4f,%.4f,%d\n",
            totalThreads, blockSize, numStreams,
            r_seq->total_ms, r_conc->total_ms,
            speedup, r_conc->data_bytes);
    fclose(csv);
}

int main(int argc, char **argv) {
    if (argc < 3) { print_usage(argv[0]); return 1; }

    int totalThreads = atoi(argv[1]);
    int blockSize = atoi(argv[2]);
    int numStreams = NUM_STREAMS_DEFAULT;
    const char *filePath = NULL;
    const char *csvPath = NULL;

    // Parse optional flags
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--file") == 0 && i + 1 < argc) {
            filePath = argv[++i];
        } else if (strcmp(argv[i], "--streams") == 0 && i + 1 < argc) {
            numStreams = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            csvPath = argv[++i];
        }
    }

    if (!filePath) {
        fprintf(stderr, "Error: specify --file PATH\n");
        print_usage(argv[0]);
        return 1;
    }

    int numBlocks = (totalThreads + blockSize - 1) / blockSize;
    print_config(totalThreads, blockSize, numBlocks, numStreams, filePath);

    // Read input text file
    FILE *fp = fopen(filePath, "r");
    if (!fp) {
        fprintf(stderr, "Error: cannot open %s\n", filePath);
        return 1;
    }
    int text_size;
    unsigned char *text = read_all(fp, &text_size);
    fclose(fp);
    printf("  text bytes   : %d\n\n", text_size);

    // Warmup
    PipelineResult warmup;
    run_pipeline(text, text_size < 1024 ? text_size : 1024,
                 1, 64, 64, &warmup);
    free(warmup.streams);

    // Run 1: Sequential (1 stream)
    printf("\nRun 1: Sequential (1 stream)\n");
    PipelineResult r_seq;
    run_pipeline(text, text_size, 1, blockSize, totalThreads, &r_seq);
    print_stream_table(&r_seq);

    // Run 2: Concurrent (N streams)
    printf("\nRun 2: Concurrent (%d streams)\n", numStreams);
    PipelineResult r_conc;
    run_pipeline(text, text_size, numStreams, blockSize, totalThreads,
                 &r_conc);
    print_stream_table(&r_conc);

    print_comparison(&r_seq, &r_conc);
    if (csvPath) write_csv_row(csvPath, totalThreads, blockSize,
                               numStreams, &r_seq, &r_conc);

    // Cleanup
    free(r_seq.streams);
    free(r_conc.streams);
    free(text);
    return 0;
}
