// mandelbrot.cpp — OpenCL Mandelbrot set generator with zoom animation

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>

#include "info.hpp"

// ============================================================================
// Constants
// ============================================================================

#define LOCAL_SIZE_X  16
#define LOCAL_SIZE_Y  16
#define NUM_STRIPS    4

// ============================================================================
// Types
// ============================================================================

struct Config {
    int    width    = 1920;
    int    height   = 1080;
    int    maxIter  = 256;
    float  cx       = -0.5f;
    float  cy       = 0.0f;
    float  zoom     = 1.0f;
    float  zoomEnd  = 0.0f;
    int    platform = 0;
    int    frames   = 1;
    bool   runCPU   = false;
    std::string outFile = "mandelbrot.ppm";
};

struct CLState {
    cl_context       context;
    cl_command_queue  queue;
    cl_program        program;
    cl_kernel         kernel;
    cl_device_id      deviceID;
    cl_mem            mainBuffer;
    std::vector<cl_mem> subBuffers;
    std::vector<int>    stripOffsets;
    std::vector<int>    stripHeights;
};

// ============================================================================
// Helpers
// ============================================================================

inline void checkErr(cl_int err, const char *name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name
                  << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

static std::string readKernelSource(
    const char *filename)
{
    std::ifstream srcFile(filename);
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1,
             filename);
    return std::string(
        std::istreambuf_iterator<char>(srcFile),
        std::istreambuf_iterator<char>());
}

static size_t roundUp(size_t value, size_t multiple)
{
    size_t rem = value % multiple;
    if (rem == 0) return value;
    return value + multiple - rem;
}

// ============================================================================
// PPM output
// ============================================================================

static void writePPM(
    const char *filename,
    const float *data,
    int width,
    int height,
    int maxIter)
{
    std::ofstream ofs(filename, std::ios::binary);
    ofs << "P6\n" << width << " "
        << height << "\n255\n";

    for (int i = 0; i < width * height; i++) {
        float val = data[i];
        unsigned char r, g, b;
        if (val >= (float)maxIter) {
            r = g = b = 0;
        } else {
            float t = val / (float)maxIter;
            float hue = 360.0f * t;
            float c = 1.0f;
            float x = c * (1.0f -
                fabsf(fmodf(hue / 60.0f, 2.0f)
                      - 1.0f));
            float m = 0.0f;
            float rf, gf, bf;
            if (hue < 60.0f) {
                rf = c; gf = x; bf = 0;
            } else if (hue < 120.0f) {
                rf = x; gf = c; bf = 0;
            } else if (hue < 180.0f) {
                rf = 0; gf = c; bf = x;
            } else if (hue < 240.0f) {
                rf = 0; gf = x; bf = c;
            } else if (hue < 300.0f) {
                rf = x; gf = 0; bf = c;
            } else {
                rf = c; gf = 0; bf = x;
            }
            r = (unsigned char)((rf + m) * 255.0f);
            g = (unsigned char)((gf + m) * 255.0f);
            b = (unsigned char)((bf + m) * 255.0f);
        }
        ofs.put(r);
        ofs.put(g);
        ofs.put(b);
    }
    ofs.close();
}

// ============================================================================
// CPU reference
// ============================================================================

static void mandelbrotCPU(
    float *output,
    int width,
    int height,
    float xMin,
    float xMax,
    float yMin,
    float yMax,
    int maxIter)
{
    for (int py = 0; py < height; py++) {
        for (int px = 0; px < width; px++) {
            float cr = xMin + px * (xMax - xMin)
                       / (float)width;
            float ci = yMin + py * (yMax - yMin)
                       / (float)height;
            float zr = 0.0f, zi = 0.0f;
            int iter = 0;
            while (iter < maxIter) {
                float zr2 = zr * zr;
                float zi2 = zi * zi;
                if (zr2 + zi2 > 4.0f) break;
                zi = 2.0f * zr * zi + ci;
                zr = zr2 - zi2 + cr;
                iter++;
            }
            float val;
            if (iter == maxIter) {
                val = (float)maxIter;
            } else {
                float mag2 = zr * zr + zi * zi;
                val = (float)iter + 1.0f
                    - log2f(log2f(mag2) * 0.5f);
            }
            output[py * width + px] = val;
        }
    }
}

// ============================================================================
// CLI parsing
// ============================================================================

static Config parseArgs(int argc, char **argv)
{
    Config cfg;
    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--width" && i + 1 < argc)
            cfg.width = atoi(argv[++i]);
        else if (arg == "--height" && i + 1 < argc)
            cfg.height = atoi(argv[++i]);
        else if (arg == "--iter" && i + 1 < argc)
            cfg.maxIter = atoi(argv[++i]);
        else if (arg == "--cx" && i + 1 < argc)
            cfg.cx = (float)atof(argv[++i]);
        else if (arg == "--cy" && i + 1 < argc)
            cfg.cy = (float)atof(argv[++i]);
        else if (arg == "--zoom" && i + 1 < argc)
            cfg.zoom = (float)atof(argv[++i]);
        else if (arg == "--platform" && i + 1 < argc)
            cfg.platform = atoi(argv[++i]);
        else if (arg == "--frames" && i + 1 < argc)
            cfg.frames = atoi(argv[++i]);
        else if (arg == "--zoom-end" && i + 1 < argc)
            cfg.zoomEnd = (float)atof(argv[++i]);
        else if (arg == "--cpu")
            cfg.runCPU = true;
        else if (arg == "--output" && i + 1 < argc)
            cfg.outFile = argv[++i];
    }
    if (cfg.zoomEnd <= 0.0f) cfg.zoomEnd = cfg.zoom;
    return cfg;
}

// ============================================================================
// OpenCL setup and teardown
// ============================================================================

static CLState initOpenCL(const Config &cfg)
{
    CLState cl;
    cl_int errNum;
    cl_uint numPlatforms;

    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr(
        (errNum != CL_SUCCESS) ? errNum :
        (numPlatforms <= 0 ? -1 : CL_SUCCESS),
        "clGetPlatformIDs");

    std::vector<cl_platform_id> platformIDs(numPlatforms);
    errNum = clGetPlatformIDs(
        numPlatforms, platformIDs.data(), NULL);
    checkErr(errNum, "clGetPlatformIDs");

    cl_uint numDevices;
    errNum = clGetDeviceIDs(
        platformIDs[cfg.platform], CL_DEVICE_TYPE_ALL,
        0, NULL, &numDevices);
    checkErr(errNum, "clGetDeviceIDs");

    std::vector<cl_device_id> deviceIDs(numDevices);
    errNum = clGetDeviceIDs(
        platformIDs[cfg.platform], CL_DEVICE_TYPE_ALL,
        numDevices, deviceIDs.data(), NULL);
    checkErr(errNum, "clGetDeviceIDs");
    cl.deviceID = deviceIDs[0];

    cl_context_properties ctxProps[] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIDs[cfg.platform],
        0
    };
    cl.context = clCreateContext(
        ctxProps, 1, &cl.deviceID,
        NULL, NULL, &errNum);
    checkErr(errNum, "clCreateContext");

    cl.queue = clCreateCommandQueue(
        cl.context, cl.deviceID,
        CL_QUEUE_PROFILING_ENABLE, &errNum);
    checkErr(errNum, "clCreateCommandQueue");

    std::string src = readKernelSource("mandelbrot.cl");
    const char *srcPtr = src.c_str();
    size_t srcLen = src.length();

    cl.program = clCreateProgramWithSource(
        cl.context, 1, &srcPtr, &srcLen, &errNum);
    checkErr(errNum, "clCreateProgramWithSource");

    errNum = clBuildProgram(
        cl.program, 1, &cl.deviceID,
        "-cl-fast-relaxed-math", NULL, NULL);
    if (errNum != CL_SUCCESS) {
        char buildLog[16384];
        clGetProgramBuildInfo(cl.program, cl.deviceID,
            CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog), buildLog, NULL);
        std::cerr << "Build error:\n"
                  << buildLog << std::endl;
        checkErr(errNum, "clBuildProgram");
    }

    cl.kernel = clCreateKernel(
        cl.program, "mandelbrot_smooth", &errNum);
    checkErr(errNum, "clCreateKernel");

    return cl;
}

static void createSubBuffers(
    CLState &cl, int width, int height)
{
    cl_int errNum;
    size_t totalPixels = (size_t)width * height;
    size_t bufSize = totalPixels * sizeof(float);

    cl.mainBuffer = clCreateBuffer(
        cl.context, CL_MEM_WRITE_ONLY,
        bufSize, NULL, &errNum);
    checkErr(errNum, "clCreateBuffer(main)");

    int stripHeight = height / NUM_STRIPS;
    int remainder   = height % NUM_STRIPS;

    for (int s = 0; s < NUM_STRIPS; s++) {
        int sH = stripHeight +
                 (s < remainder ? 1 : 0);
        int sOff = 0;
        for (int j = 0; j < s; j++)
            sOff += stripHeight +
                    (j < remainder ? 1 : 0);

        cl.stripOffsets.push_back(sOff);
        cl.stripHeights.push_back(sH);

        cl_buffer_region region;
        region.origin =
            (size_t)sOff * width * sizeof(float);
        region.size =
            (size_t)sH * width * sizeof(float);

        cl_mem sub = clCreateSubBuffer(
            cl.mainBuffer, CL_MEM_WRITE_ONLY,
            CL_BUFFER_CREATE_TYPE_REGION,
            &region, &errNum);
        checkErr(errNum, "clCreateSubBuffer");
        cl.subBuffers.push_back(sub);
    }
}

static void releaseOpenCL(CLState &cl)
{
    for (auto &sub : cl.subBuffers)
        clReleaseMemObject(sub);
    clReleaseMemObject(cl.mainBuffer);
    clReleaseKernel(cl.kernel);
    clReleaseProgram(cl.program);
    clReleaseCommandQueue(cl.queue);
    clReleaseContext(cl.context);
}

// ============================================================================
// Rendering
// ============================================================================

static void renderFrame(
    CLState &cl,
    float *hostOutput,
    const Config &cfg,
    float curZoom,
    const std::string &filename)
{
    cl_int errNum;
    float aspect = (float)cfg.width / (float)cfg.height;
    float viewH = 2.0f / curZoom;
    float viewW = viewH * aspect;
    float xMin = cfg.cx - viewW / 2.0f;
    float xMax = cfg.cx + viewW / 2.0f;
    float yMin = cfg.cy - viewH / 2.0f;
    float yMax = cfg.cy + viewH / 2.0f;

    size_t totalPixels =
        (size_t)cfg.width * cfg.height;
    size_t bufSize = totalPixels * sizeof(float);

    std::vector<cl_event> events;
    for (int s = 0; s < NUM_STRIPS; s++) {
        int sH   = cl.stripHeights[s];
        int sOff = cl.stripOffsets[s];
        float sYMin = yMin + sOff
            * (yMax - yMin) / (float)cfg.height;
        float sYMax = yMin + (sOff + sH)
            * (yMax - yMin) / (float)cfg.height;

        int w = cfg.width;
        int mi = cfg.maxIter;
        errNum  = clSetKernelArg(cl.kernel, 0,
            sizeof(cl_mem), &cl.subBuffers[s]);
        errNum |= clSetKernelArg(cl.kernel, 1,
            sizeof(int), &w);
        errNum |= clSetKernelArg(cl.kernel, 2,
            sizeof(int), &sH);
        errNum |= clSetKernelArg(cl.kernel, 3,
            sizeof(float), &xMin);
        errNum |= clSetKernelArg(cl.kernel, 4,
            sizeof(float), &xMax);
        errNum |= clSetKernelArg(cl.kernel, 5,
            sizeof(float), &sYMin);
        errNum |= clSetKernelArg(cl.kernel, 6,
            sizeof(float), &sYMax);
        errNum |= clSetKernelArg(cl.kernel, 7,
            sizeof(int), &mi);
        checkErr(errNum, "clSetKernelArg");

        size_t globalSize[2] = {
            roundUp(cfg.width, LOCAL_SIZE_X),
            roundUp(sH, LOCAL_SIZE_Y)
        };
        size_t localSize[2] = {
            LOCAL_SIZE_X, LOCAL_SIZE_Y
        };

        cl_event evt;
        errNum = clEnqueueNDRangeKernel(
            cl.queue, cl.kernel, 2, NULL,
            globalSize, localSize,
            0, NULL, &evt);
        checkErr(errNum, "clEnqueueNDRangeKernel");
        events.push_back(evt);
    }

    clWaitForEvents(events.size(), events.data());

    float *mapped = (float *)clEnqueueMapBuffer(
        cl.queue, cl.mainBuffer, CL_TRUE,
        CL_MAP_READ, 0, bufSize,
        0, NULL, NULL, &errNum);
    checkErr(errNum, "clEnqueueMapBuffer");
    memcpy(hostOutput, mapped, bufSize);
    errNum = clEnqueueUnmapMemObject(
        cl.queue, cl.mainBuffer, mapped,
        0, NULL, NULL);
    checkErr(errNum, "clEnqueueUnmapMemObject");
    clFinish(cl.queue);

    for (auto &evt : events)
        clReleaseEvent(evt);

    writePPM(filename.c_str(), hostOutput,
             cfg.width, cfg.height, cfg.maxIter);
}

static void runCPUComparison(const Config &cfg)
{
    float aspect = (float)cfg.width / (float)cfg.height;
    float viewH = 2.0f / cfg.zoom;
    float viewW = viewH * aspect;
    float xMin = cfg.cx - viewW / 2.0f;
    float xMax = cfg.cx + viewW / 2.0f;
    float yMin = cfg.cy - viewH / 2.0f;
    float yMax = cfg.cy + viewH / 2.0f;

    size_t totalPixels =
        (size_t)cfg.width * cfg.height;
    float *cpuOutput = new float[totalPixels];
    mandelbrotCPU(cpuOutput, cfg.width, cfg.height,
                  xMin, xMax, yMin, yMax, cfg.maxIter);
    writePPM("mandelbrot_cpu.ppm", cpuOutput,
             cfg.width, cfg.height, cfg.maxIter);
    delete[] cpuOutput;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv)
{
    Config cfg = parseArgs(argc, argv);
    bool isAnimation = (cfg.frames > 1);

    CLState cl = initOpenCL(cfg);
    createSubBuffers(cl, cfg.width, cfg.height);

    size_t totalPixels =
        (size_t)cfg.width * cfg.height;
    float *hostOutput = new float[totalPixels];

    if (isAnimation) {
        if (system("mkdir -p frames") != 0) {
            std::cerr << "ERROR: mkdir frames" << std::endl;
            return EXIT_FAILURE;
        }
    }

    for (int f = 0; f < cfg.frames; f++) {
        float t = (cfg.frames > 1)
            ? (float)f / (float)(cfg.frames - 1)
            : 0.0f;
        float curZoom = cfg.zoom *
            powf(cfg.zoomEnd / cfg.zoom, t);

        std::string filename;
        if (isAnimation) {
            std::ostringstream oss;
            oss << "frames/frame_"
                << std::setw(5) << std::setfill('0')
                << f << ".ppm";
            filename = oss.str();
        } else {
            filename = cfg.outFile;
        }

        renderFrame(cl, hostOutput, cfg,
                    curZoom, filename);
    }

    if (cfg.runCPU && !isAnimation)
        runCPUComparison(cfg);

    delete[] hostOutput;
    releaseOpenCL(cl);
    return 0;
}
