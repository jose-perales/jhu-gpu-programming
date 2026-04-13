// mandelbrot.cl — Mandelbrot set kernels using float2 complex arithmetic

// Integer iteration count (escape-time algorithm: z = z^2 + c)
__kernel void mandelbrot(
    __global int *output,
    const int width,
    const int height,
    const float x_min,
    const float x_max,
    const float y_min,
    const float y_max,
    const int max_iter)
{
    int px = get_global_id(0);
    int py = get_global_id(1);
    if (px >= width || py >= height) return;

    // Map pixel to complex plane using float2 vector
    float2 c = (float2)(
        x_min + px * (x_max - x_min) / (float)width,
        y_min + py * (y_max - y_min) / (float)height
    );

    float2 z = (float2)(0.0f, 0.0f);
    int iter = 0;

    // Iterate z = z^2 + c until escape or max
    while (iter < max_iter) {
        float zr2 = z.x * z.x;
        float zi2 = z.y * z.y;
        if (zr2 + zi2 > 4.0f) break;
        z = (float2)(zr2 - zi2 + c.x,
                     2.0f * z.x * z.y + c.y);
        iter++;
    }

    output[py * width + px] = iter;
}

// Smooth (continuous) coloring variant for gradient output
__kernel void mandelbrot_smooth(
    __global float *output,
    const int width,
    const int height,
    const float x_min,
    const float x_max,
    const float y_min,
    const float y_max,
    const int max_iter)
{
    int px = get_global_id(0);
    int py = get_global_id(1);
    if (px >= width || py >= height) return;

    float2 c = (float2)(
        x_min + px * (x_max - x_min) / (float)width,
        y_min + py * (y_max - y_min) / (float)height
    );

    float2 z = (float2)(0.0f, 0.0f);
    int iter = 0;

    while (iter < max_iter) {
        float zr2 = z.x * z.x;
        float zi2 = z.y * z.y;
        if (zr2 + zi2 > 4.0f) break;
        z = (float2)(zr2 - zi2 + c.x,
                     2.0f * z.x * z.y + c.y);
        iter++;
    }

    // Smooth coloring: use log-log escape value
    float smooth_val;
    if (iter == max_iter) {
        smooth_val = (float)max_iter;
    } else {
        float mag2 = z.x * z.x + z.y * z.y;
        smooth_val = (float)iter +
            1.0f - log2(log2(mag2) * 0.5f);
    }

    output[py * width + px] = smooth_val;
}
