#include <Windows.h>

#include <stdio.h>
#include <math.h>
#include <gl/glew.h>
#include <opencv2/opencv.hpp>
#include <string>
#include "vec.h"
#include "gpe.h"

#pragma comment(lib,"opengl32.lib")
#pragma comment(lib,"glu32.lib")
#pragma comment(lib,"glew32.lib")
#pragma comment(lib,"cufft.lib")

#ifdef _DEBUG
#pragma comment(lib,"opencv_world411d.lib")
#else
#pragma comment(lib,"opencv_world411.lib")
#endif 

#define MAX_LOADSTRING 100
#define PI 3.14159265358979324

HINSTANCE hInst;                                // current instance
WCHAR szTitle[MAX_LOADSTRING] = L"gpe";                  // The title bar text
WCHAR szWindowClass[MAX_LOADSTRING] = L"w32";            // the main window class name

HDC hdc1, hdc2;
HGLRC m_hrc;
int mx, my, cx, cy;
double ang1, ang2, len, cenx, ceny, cenz;

int type = 0;
float3* buffer, * cbuffer;
float* fbuffer;
current2* u;
current* v;

struct conf_t {
	std::string init, dir;
	int size;
	float vrange;
	float power;
	float g, l, r;
	int show, record;
};

conf_t conf;

int fileindex(const std::string& dir) {
	FILE* fi;
	int index = 0;
	if (!fopen_s(&fi, (dir + "/index.txt").c_str(), "rb")) {
		fscanf(fi, "%d", &index);
		fclose(fi);
	}
	if (!fopen_s(&fi, (dir + "/index.txt").c_str(), "wb")) {
		fprintf(fi, "%d", index + 1);
		fclose(fi);
	}
	return index;
}


ATOM                MyRegisterClass(HINSTANCE hInstance);
BOOL                InitInstance(HINSTANCE, int);
LRESULT CALLBACK    WndProc(HWND, UINT, WPARAM, LPARAM);
int cuevo(int size, current dt, current g, current r, current l);
int cuinit(int size);
int getu(void* u, int size,int type);
int setu(void* u, int size);
int setpotantial(void* u, int size);
int setpower(void* u, int size); 
int setnr(void* nr, int size);

int draw(float3* buffer, float3* colorbuffer) {
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear(0x00004100);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(len * cos(ang1) * cos(ang2) + cenx, len * sin(ang2) + ceny, len * sin(ang1) * cos(ang2) + cenz, cenx, ceny, cenz, 0, cos(ang2), 0);

	glBindTexture(GL_TEXTURE_2D, 0);
	glBegin(GL_LINES);
	glColor3f(1.0, 1.0, 0);
	glVertex3f(1.0f, 0.0f, -0.0f);
	glVertex3f(-1.0f, 0.0f, -0.0f);
	glVertex3f(0.0f, 1.0f, -0.0f);
	glVertex3f(0.0f, -1.0f, -0.0f);
	glVertex3f(0.0f, 0.0f, 1.0f);
	glVertex3f(0.0f, 0.0f, -1.0f);
	glEnd();

	glColor3f(0.0, 0.0, 1.0);
	glVertexPointer(3, GL_FLOAT, 0, buffer);
	glColorPointer(3, GL_FLOAT, 0, colorbuffer);
	glDrawArrays(GL_LINES, 0, conf.size * (conf.size - 1) * 4);

	SwapBuffers(wglGetCurrentDC());
	return 0;
}


float3 phasetocolor(current2 x) {
	float n;
	n = norm(x);
	n = 1.0f / sqrt(n);
	float r, g, b;
	r = 0.5f * (x.x * n + 1.0);
	g = 0.5f * ((-x.x * 0.5f + x.y * 0.865f) * n + 1.0f);
	b = 0.5f * ((-x.x * 0.5f - x.y * 0.865f) * n + 1.0f);
	return { r,g,b};
}

cv::Mat ctof(cv::Mat c) {
	cv::Mat f(c.rows, c.cols, CV_32FC1, cv::Scalar(0.0));
	for (int i = 0; i < c.rows * c.cols; i++) {
		((float*)(f.data))[i] = c.data[i * 3] * (1 / 256.0) + c.data[i * 3 + 1] * (1 / 65536.0) + c.data[i * 3 + 2] * (1 / 16777216.0);
	}
	return f;
}

int init(conf_t conf) {
	int i, j;
	cv::Mat mat(conf.size, conf.size, CV_32FC1, cv::Scalar(0));

	if (conf.init.length()) {
		FILE* fi;
		if (!fopen_s(&fi, conf.init.c_str(), "rb")) {
			fread(fbuffer, 1, conf.size * conf.size * sizeof(float) * 3, fi);
			fclose(fi);
		}
		for (i = 0; i < conf.size; i++) {
			for (j = 0; j < conf.size; j++) {
				u[i * conf.size + j] = (2.0 + j * (0.0 / conf.size)) * current2 { fbuffer[(i * conf.size + j) * 2], fbuffer[(i * conf.size + j) * 2 + 1] };
				v[i * conf.size + j] = 0.0 * fbuffer[i * conf.size + j + conf.size * conf.size * 2];
			}
		}
	}
	else {
		for (i = 0; i < conf.size; i++) {
			for (j = 0; j < conf.size; j++) {
				current r2;
				current y = (i + 0.5) / conf.size;
				current x = (j + 0.5) / conf.size;
				r2 = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5);
				r2 = r2 * 2048.0f;
				u[i * conf.size + j] = { exp(-r2) * 32.0f * (x - 0.52f),0.0f };
				v[i * conf.size + j] = 0.0f;
			}
		}
	}
	setu(u, conf.size);
	setnr(v, conf.size);

	cv::resize(ctof(cv::imread("D:/files/data/bo/v_te_16_large.png")), mat, cv::Size(conf.size, conf.size));
	for (int i = 0; i < conf.size; i++) {
		for (int j = 0; j < conf.size; j++) {
			//v[i * size + j] = ((float*)(mat.data))[(size - i - 1) * size + j]*0.25*40*40*40*40-110000;
			v[i * conf.size + j] = ((float*)(mat.data))[(conf.size - i - 1) * conf.size + j] * 29177.0 - 29177.0;
		}
	}
	setpotantial(v, conf.size);
	cv::resize(ctof(cv::imread("D:/files/data/bo/p_te.png")), mat, cv::Size(conf.size, conf.size));
	for (int i = 0; i < conf.size; i++) {
		for (int j = 0; j < conf.size; j++) {
			v[i * conf.size + j] = ((float*)(mat.data))[(conf.size - i - 1) * conf.size + j]*16.0;
		}
	}
	setpower(v, conf.size);
	return 0;
}

int main(int argc, char** argv) {
	int i, j;
	char s[256];
	unsigned int t, t1, count, f;
	MSG msg;
	HINSTANCE hInstance = ::GetModuleHandle(NULL);

	MyRegisterClass(hInstance);
	if (!InitInstance(hInstance, SW_SHOW))
	{
		return FALSE;
	}

	t = GetTickCount();
	count = 0;
	f = 0;

	conf.init = "D:/files/data/bo/record/init2048_te.data";
	conf.dir = "d:/files/data/bo/te_2048";
	conf.size = 2048;
	conf.record = 1;

	u = (current2*)malloc(conf.size * conf.size * sizeof(current2));
	v = (current*)malloc(conf.size * conf.size * sizeof(current));
	fbuffer = (float*)malloc(conf.size * conf.size * sizeof(float) * 3);
	buffer = (float3*)malloc(conf.size * (conf.size - 1) * 4 * sizeof(float3));
	cbuffer = (float3*)malloc(conf.size * (conf.size - 1) * 4 * sizeof(float3));

	cuinit(conf.size);
	init(conf);


	cv::VideoWriter writer;
//	writer.open("C:/files/tr4_2x.mp4", cv::VideoWriter::fourcc('a', 'v', 'c', '1'), 30.0, cv::Size(size, size));
	cv::Mat img(conf.size, conf.size, CV_8UC3, cv::Scalar(0, 0, 0));
	//cv::Mat img2(size*2, size*2, CV_8UC3, cv::Scalar(0, 0, 0));
	int frame = 0;
	float inn;
	// Main message loop:	
st:
	for (;;) {
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
			if (msg.message == WM_QUIT)
				break;
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		else {
			for (i = 0; i < 1024; i++) {
				cuevo(conf.size, 1.0/512.0/512.0/4.0/16.0,512.0,102400.0,384.0);
			}
			//record ever 1/16384
			frame++;
		//	printf("%f\n", frame / 4800.0);
		//	for (int i = 0; i < size * size; i++) {
		//		float x = log(norm(u[i]) * 5*20.0+1.0)*0.3*256.0;
		//		float3 color = phasetocolor(u[i]);
		//		img.data[i * 3] = x*color.z>255.0?255.0: x * color.z;
		//		img.data[i * 3 + 1] = x * color.y > 255.0 ? 255.0 : x * color.y;
		//		img.data[i * 3 + 2] = x * color.x > 255.0 ? 255.0 : x * color.x;
		//	}
		//	cv::resize(img, img2, cv::Size(size * 2, size * 2), 2.0, 2.0, cv::INTER_NEAREST);
		//	writer << img;
			if (frame >= 2050) {
				writer.release();
				return 0;
			}

			if (conf.show) {
				for (i = 0; i < conf.size; i++) {
					for (j = 0; j < conf.size - 1; j++) {
						buffer[i * (conf.size - 1) * 2 + j * 2] = { (j + 0.5f) / conf.size,(float)(4 * norm(u[i * conf.size + j])),-(i + 0.5f) / conf.size };
						buffer[i * (conf.size - 1) * 2 + j * 2 + 1] = { (j + 1.5f) / conf.size,(float)(4 * norm(u[i * conf.size + j + 1])),-(i + 0.5f) / conf.size };
						buffer[i * (conf.size - 1) * 2 + j * 2 + conf.size * (conf.size - 1) * 2] = { (i + 0.5f) / conf.size, (float)(4 * norm(u[j * conf.size + i])) ,-(j + 0.5f) / conf.size };
						buffer[i * (conf.size - 1) * 2 + j * 2 + conf.size * (conf.size - 1) * 2 + 1] = { (i + 0.5f) / conf.size,(float)(4 * norm(u[(j + 1) * conf.size + i])),-(j + 1.5f) / conf.size };
						cbuffer[i * (conf.size - 1) * 2 + j * 2] = phasetocolor(u[i * conf.size + j]);
						cbuffer[i * (conf.size - 1) * 2 + j * 2 + 1] = phasetocolor(u[i * conf.size + j + 1]);
						cbuffer[i * (conf.size - 1) * 2 + j * 2 + conf.size * (conf.size - 1) * 2] = phasetocolor(u[j * conf.size + i]);
						cbuffer[i * (conf.size - 1) * 2 + j * 2 + conf.size * (conf.size - 1) * 2 + 1] = phasetocolor(u[(j + 1) * conf.size + i]);
					}
				}

				inn = 0;

				getu(u, conf.size, type);
				for (i = 0; i < conf.size * conf.size; i++) {
					inn += norm(u[i]);
				}
				printf("%fns,%f\n", frame * 256.0 * 3.841 * 1.0 / 512.0 / 512.0 / 4.0, inn);
				if (type) {
					for (i = conf.size * conf.size - 1; i >= 0; i--) {
						((current*)u)[i * 2] = sqrt(((current*)u)[i]);
						((current*)u)[i * 2 + 1] = ((current*)u)[i * 2];
					}
				}
				draw(buffer, cbuffer);
			}
			
			if (conf.record && frame > 1024) {
				char info[1024];
				int index = fileindex(conf.dir);
				sprintf_s(info, "%s/%d.data", conf.dir.c_str(), index);
				FILE* fi;
				if (!fopen_s(&fi, info, "wb")) {
					getu(u, conf.size, 0);
					getu(v, conf.size, 1);
					for (int i = 0; i < conf.size * conf.size; i++) {
						fbuffer[i * 2] = u[i].x;
						fbuffer[i * 2 + 1] = u[i].y;
						fbuffer[i + conf.size * conf.size * 2] = v[i];
					}
					fwrite(fbuffer, 1, conf.size * conf.size * sizeof(float) * 3, fi);
					fclose(fi);
				}
			}
		}
	}
	writer.release();
	return (int)msg.wParam;
}

ATOM MyRegisterClass(HINSTANCE hInstance)
{
	WNDCLASSEXW wcex;

	wcex.cbSize = sizeof(WNDCLASSEX);

	wcex.style = CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc = WndProc;
	wcex.cbClsExtra = 0;
	wcex.cbWndExtra = 0;
	wcex.hInstance = hInstance;
	wcex.hIcon = LoadIcon(hInstance, NULL);
	wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
	wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	wcex.lpszMenuName = MAKEINTRESOURCEW(NULL);
	wcex.lpszClassName = szWindowClass;
	wcex.hIconSm = LoadIcon(wcex.hInstance, NULL);

	return RegisterClassExW(&wcex);
}

BOOL InitInstance(HINSTANCE hInstance, int nCmdShow)
{
	hInst = hInstance; // Store instance handle in our global variable

	HWND hWnd = CreateWindowW(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW,
		CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, nullptr, nullptr, hInstance, nullptr);

	if (!hWnd)
	{
		return FALSE;
	}

	ShowWindow(hWnd, nCmdShow);
	UpdateWindow(hWnd);

	return TRUE;
}


LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message)
	{
	case WM_COMMAND:
	{
		int wmId = LOWORD(wParam);
		// Parse the menu selections:
		switch (wmId)
		{
		default:
			return DefWindowProc(hWnd, message, wParam, lParam);
		}
	}
	break;
	case WM_PAINT:
	{
		PAINTSTRUCT ps;
		HDC hdc = BeginPaint(hWnd, &ps);
		// TODO: Add any drawing code that uses hdc here...
		//draw();
		EndPaint(hWnd, &ps);
	}
	break;
	case WM_DESTROY: {
		PostQuitMessage(0);
		break;
	}
	case WM_CREATE: {
		int i, j;
		char s[1024];
		PIXELFORMATDESCRIPTOR pfd = {
			sizeof(PIXELFORMATDESCRIPTOR),
			1,
			PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER | PFD_STEREO,
			PFD_TYPE_RGBA,
			24,
			0,0,0,0,0,0,0,0,
			0,
			0,0,0,0,
			32,
			0,0,
			PFD_MAIN_PLANE,
			0,0,0,0
		};
		hdc1 = GetDC(hWnd);
		hdc2 = GetDC(NULL);
		int uds = ::ChoosePixelFormat(hdc1, &pfd);
		::SetPixelFormat(hdc1, uds, &pfd);
		m_hrc = ::wglCreateContext(hdc1);
		::wglMakeCurrent(hdc1, m_hrc);
		glewInit();
		glDisable(GL_CULL_FACE);
		glEnable(GL_DEPTH_TEST);
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_COLOR_ARRAY);
		glEnable(GL_TEXTURE_1D);
		glEnable(GL_TEXTURE_2D);

		ang1 = PI * 0.5;
		ang2 = 0.7;
		len = 4;
		cenx = 0.5;
		ceny = 0.0;
		cenz = -0.5;
		break;
	}
	case WM_SIZE: {
		cx = lParam & 0xffff;
		cy = (lParam & 0xffff0000) >> 16;
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glFrustum(-(float)cx / (cx + cy) * len * 0.0078125, (float)cx / (cx + cy) * len * 0.0078125, -(float)cy / (cx + cy) * len * 0.0078125, (float)cy / (cx + cy) * len * 0.0078125, len * 0.00390625 * 4, len * 400.0);
		glViewport(0, 0, cx, cy);
		break;
	}
	case WM_MOUSEMOVE: {
		int x, y, f;
		f = 0;
		x = (lParam & 0xffff);
		y = ((lParam & 0xffff0000) >> 16);
		if (MK_LBUTTON & wParam) {
			f = 1;
			ang1 += (x - mx) * 0.002;
			ang2 += (y - my) * 0.002;
		}
		if (MK_RBUTTON & wParam) {
			double l;
			f = 1;
			l = len * 1.25 / (cx + cy);
			cenx += l * (-(x - mx) * sin(ang1) - (y - my) * sin(ang2) * cos(ang1));
			ceny += l * ((y - my) * cos(ang2));
			cenz += l * ((x - mx) * cos(ang1) - (y - my) * sin(ang2) * sin(ang1));
		}
		mx = x;
		my = y;
		if (f) {
			//draw();
		}
		break;
	}
	case WM_MOUSEWHEEL: {
		short m;
		m = (wParam & 0xffff0000) >> 16;
		len *= exp(-m * 0.001);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glFrustum(-(float)cx / (cx + cy) * len * 0.0078125, (float)cx / (cx + cy) * len * 0.0078125, -(float)cy / (cx + cy) * len * 0.0078125, (float)cy / (cx + cy) * len * 0.0078125, len * 0.00390625 * 4, len * 400.0);
		break;
	}
	case WM_KEYDOWN: {
		switch (wParam) {
		case 'T': {
			type = !type;
			break;
		}
		case 'S': {
			char info[1024];
			int index = fileindex("D:/files/data/bo/record");
			sprintf_s(info, "D:/files/data/bo/record/%d.data", index);
			FILE* fi;
			if (!fopen_s(&fi, info, "wb")) {
				getu(u, conf.size, 0);
				getu(v, conf.size, 1);
				for (int i = 0; i < conf.size * conf.size; i++) {
					fbuffer[i * 2] = u[i].x;
					fbuffer[i * 2 + 1] = u[i].y;
					fbuffer[i + conf.size * conf.size * 2] = v[i];
				}
				fwrite(fbuffer, 1, conf.size * conf.size * sizeof(float) * 3, fi);
				fclose(fi);
			}
			break;
		}
		}
		break;
	}
	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
	}
	return 0;
}