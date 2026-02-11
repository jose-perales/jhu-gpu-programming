# Instructions

1. Create a program that contains the same (or similar) algorithm executing using a CPU-based method and a CUDA kernel. There should be minimal conditional branching in the method and kernel. This should use a non-trivial amount of data 1000s to millions, however much you would like to test.
2. Create a separate program (or include in the same code base from the previous step) that demonstrates the effect of conditional branching on implementing the same CPU and GPU code. You will also need to include at least one performance comparison chart and a short text file that includes your thoughts on the results.
3. The base CUDA source code file must be called assignment.cu and be housed in the module3 directory of your repository. It will need to be runnable as `assignment.exe 512 256`. You can modify the Makefile as you see fit, as long as the make command (with no arguments), builds the assignment.exe as the executable output. The assignment.cu file that I have provided in the module3 directory handles the two arguments for total number of threads and number of threads per block. Note when running the CPU algorithm you will not need to take these arguments into account as you don't have a lot of control over that.
4. You will need to include the zipped up code for the assignment and images/video/links that show your code completing all of the parts of the rubric that it is designed to complete in what is submitted for this assignment.
5. Look at the below example of a main function for the same assignment in a previous year. If the surrounding script only executes the main function once, what is good and bad about this submission? Do not fix the assignment or submit it along with your other code. Just give me a short description of your thoughts about it.

```cpp
int main(int argc,char* argv[]) {

outputCardInfo();
int blocks = 3;
int threads = 64;
if (argc == 2) {

    blocks = atoi(argv[1]);
    printf("Blocks changed to:%i\n", blocks);

} else if (argc == 3) {

    blocks = atoi(argv[1]);
    threads = atoi(argv[2]);
    printf("Blocks changed to:%i\n", blocks);
    printf("Threads changed to:%i\n", threads);

}

int a[N], b[N], c[N];
int *dev_a, *dev_b, *dev_c;
cudaMalloc((void**)&dev_a, N * sizeof(int));
cudaMalloc((void**)&dev_b, N * sizeof(int));
cudaMalloc((void**)&dev_c, N * sizeof(int));

//Populate our arrays with numbers.
for (int i = 0; i < N; i++) {

    a[i] = -i;
    b[i] = i*i;

}

cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
auto start = std::chrono::high_resolution_clock::now();
add<<<blocks,threads>>> (dev_a, dev_b, dev_c);
auto stop = std::chrono::high_resolution_clock::now();
cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);
cudaFree(dev_a);
cudaFree(dev_b);
cudaFree(dev_c);
auto startHost = std::chrono::high_resolution_clock::now();
addHost(a, b, c);
auto stopHost = std::chrono::high_resolution_clock::now();
std::cout <endl<< " Time elapsed GPU = " << std::chrono::duration_castchrono::nanoseconds>(stop - start).count() << "ns\n";
std::cout << " Time elapsed Host = " << std::chrono::duration_castchrono::nanoseconds>(stopHost - startHost).count() << "ns\n";
return 0;

}
```

## Assignment Rubric

| Criteria | Proficient | Competent | Novice | Pts |
| ---------- | ------------ | ----------- | -------- | ----- |
| Create a program that demonstrates a large number of threads, at a minimum 64 threads with one block size, performing a simple operation in a GPU kernel. | 12.5 pts - at least 64 threads in one block at a minimum | 6.25 pts - The operation runs without a variable number of threads | 0 pts | 12.5 pts |
| In the same or separate program from above execute the same algorithm as the GPU kernel but with CPU function. Make sure that it is using the same amount of data as the GPU kernel as the basis for its work. | 12.5 pts | 6.25 pts | 0 pts | 12.5 pts |
| Update the same program or create a new one to execute the GPU kernel but with at least 1 conditional branching statement. | 12.5 pts | 6.25 pts | 0 pts | 12.5 pts |
| Update the same program or create a new one to execute the CPU method but with at least 1 conditional branching statement. | 12.5 pts | 6.25 pts | 0 pts | 12.5 pts |
| Execute or program your code to use two additional numbers of threads (capture all runs of your code). | 15 to >10.0 pts - two additional configuration for number of threads based on external command line arguments | 10 to >5.0 pts - Hard coded of two additional configuration for number of threads | 5 to >0.0 pts - one additional configuration for number of threads | 15 pts |
| Execute or program your code to use two additional block sizes (capture all runs of your code). | 15 to >10.0 pts - two additional configuration for number of threads based on external command line arguments | 10 to >5.0 pts - Hard coded for two additional configuration for number of threads | 5 to >0.0 pts - one additional block size | 15 pts |
| Develop a mechanism in your code to use command line arguments to vary the number of threads and the block size. | 5 pts - varies both threads and block size | 2.5 pts - only varies either threads or block size | 0 pts | 5 pts |
| Stretch Problem | 5 pts - Identify both issues and good qualities of the code in the assignment text | 2.5 pts - Identify issues or good qualities of the code in the assignment text | 0 pts | 5 pts |
| Quality of your code, measured by use of constants, well-named variables and functions, and useful comments in code | 10 pts - exceptional code quality | 5 pts - fairly high quality code | 0 pts - decent to good code quality | 10 pts |

Total Points: 100
