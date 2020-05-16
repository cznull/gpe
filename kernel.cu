
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cufft.h"

#include <stdio.h>
#include "gpe.h"

current2* d_u = nullptr, * d_uk = nullptr;
current* d_potantial = nullptr, * d_nr = nullptr, * d_power = nullptr;
current t = 0.0f;

cufftHandle fftPlan;
cufftResult fresu;

int cuinit(int size);

__global__ void evok2(current2* uk, int size, current dt, current halfinvereffm, current k2, current ns) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	current kx, ky, h, temp, c, s;
	kx = x > size / 2 ? x - size : x;
	ky = y > size / 2 ? y - size : y;
	h = (kx * kx + ky * ky) * k2 * halfinvereffm;;
	current2 u = uk[y * size + x];
	c = cos(h * dt);
	s = sin(h * dt);
//	c = rsqrt(1 + h * h * dt * dt);
//	s = c * h * dt;
	temp = u.x *c + u.y * s;
	u.y = (u.y * c - u.x * s) * ns;
	u.x = temp*ns;
	uk[y * size + x] = u;
}

//ns 傅里叶变换归一化
//k2 

__global__ void evov(current2* uk, int size, current* v, current* nr, current* power, current g, current r,current l,current dt,current t) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int index = y * size + x;
	current h, temp, c, s, ex, n, nrl, ledge;
	current2 u = uk[index];
	nrl = nr[index];
	n = u.x * u.x + u.y * u.y;
	h = v[index] + g * n + current(2.0) * g * nrl;
//	current mu = 0.74f * 40.0f * 40.0f;
//	current x0,y0;
//	x0 = 0.125f * cos(mu * t) * (1.0f - sin(mu * t));
//	y0 = 0.125f * cos(mu * t) * sin(mu * t);
//	h = h + 100.0f * 40.0f * 40.0f * exp(-(((x + 0.0f) / size - 0.5f - x0) * ((x + 0.0f) / size - 0.5f - x0) + ((y + 0.0f) / size - 0.5f - y0) * ((y + 0.0f) / size - 0.5f - y0)) * 17777.8f);
	c = cos(h * dt);
	s = sin(h * dt);
	ledge = max(0.0f, max(64.0f - x, 1.0f + 64.0f + x - size));
	ledge = max(ledge, max(64.0f - y, 1.0f + 64.0f + y - size));
	ex = current(0.5) * (r * nrl - l - l * ledge * 0.5);
	ex = exp(ex * dt);
//	ex = 1.0 + ex * dt;
	temp = u.x * c + u.y * s;
	u.y = u.y * c - u.x * s;
	u.y = u.y * ex;
	u.x = temp * ex;
	nrl = nrl + power[index]*dt;
	nrl = nrl * exp((-l - n * r) * dt);
	nr[index] = nrl;
	uk[index] = u;
}

int cuevo(int size, current dt, current g, current r, current l) {
	cufftExec(fftPlan, d_u, d_u, CUFFT_FORWARD);
	evok2 << < dim3(size / 128, size, 1), dim3(128, 1, 1) >> > (d_u, size, dt, 0.5, 6.2832 * 6.2832, 1.0 / size / size);
	cufftExec(fftPlan, d_u, d_u, CUFFT_INVERSE);
	evov << < dim3(size / 128, size, 1), dim3(128, 1, 1) >> > (d_u, size, d_potantial, d_nr, d_power, g, r, l, dt, t);
	t += dt;
	return 0;
}

int getu(void* u, int size, int type) {
	switch (type) {
	case 0:
		cudaMemcpy(u, d_u, size * size * sizeof(current2), cudaMemcpyDeviceToHost);
		break;
	case 1:
		cudaMemcpy(u, d_nr, size * size * sizeof(current), cudaMemcpyDeviceToHost);
		break;
	default:
		break;
	}
	return 0;
}

int setu(void* u, int size) {
	cudaMemcpy(d_u, u, size * size * sizeof(current2), cudaMemcpyHostToDevice);
	return 0;
}

int setnr(void* nr, int size) {
	cudaMemcpy(d_nr, nr, size * size * sizeof(current), cudaMemcpyHostToDevice);
	return 0;
}

int setpotantial(void* u, int size) {
	cudaMemcpy(d_potantial, u, size * size * sizeof(current), cudaMemcpyHostToDevice);
	return 0;
}

int setpower(void* u, int size) {
	cudaMemcpy(d_power, u, size * size * sizeof(current), cudaMemcpyHostToDevice);
	return 0;
}


int cuinit(int size) {
	cudaMalloc(&d_potantial, size * size * sizeof(current));
	cudaMalloc(&d_nr, size * size * sizeof(current));
	cudaMemset(d_nr, 0, size * size * sizeof(current));
	cudaMalloc(&d_power, size * size * sizeof(current));
	cudaMalloc(&d_u, size * size * sizeof(current2));
	cudaMalloc(&d_uk, size * size * sizeof(current2));
	cufftPlan2d(&fftPlan, size, size, CUFFT_cur2cur);
	return 0;
}