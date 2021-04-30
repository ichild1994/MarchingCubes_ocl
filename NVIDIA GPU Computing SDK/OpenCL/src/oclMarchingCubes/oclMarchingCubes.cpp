/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* 
    Marching cubes

    This sample extracts a geometric isosurface from a volume dataset using
    the marching cubes algorithm. It uses the scan (prefix sum) function 
    from the SDK sample oclScan to perform stream compaction. Similar techniques can
    be used for other problems that require a variable-sized output per
    thread.

    For more information on marching cubes see:
    http://local.wasp.uwa.edu.au/~pbourke/geometry/polygonise/
    http://en.wikipedia.org/wiki/Marching_cubes

    Volume data courtesy:
    http://www9.informatik.uni-erlangen.de/External/vollib/


    The algorithm consists of several stages:

    1. Execute "classifyVoxel" kernel
    This evaluates the volume at the corners of each voxel and computes the
    number of vertices each voxel will generate.
    It is executed using one thread per voxel.
    It writes two arrays - voxelOccupied and voxelVertices to global memory.
    voxelOccupied is a flag indicating if the voxel is non-empty.

    2. Scan "voxelOccupied" array 
    Read back the total number of occupied voxels from GPU to CPU.
    This is the sum of the last value of the exclusive scan and the last
    input value.

    3. Execute "compactVoxels" kernel
    This compacts the voxelOccupied array to get rid of empty voxels.
    This allows us to run the complex "generateTriangles" kernel on only
    the occupied voxels.

    4. Scan voxelVertices array
    This gives the start address for the vertex data for each voxel.
    We read back the total number of vertices generated from GPU to CPU.

    Note that by using a custom scan function we could combine the above two
    scan operations above into a single operation.

    5. Execute "generateTriangles" kernel
    This runs only on the occupied voxels.
    It looks up the field values again and generates the triangle data,
    using the results of the scan to write the output to the correct addresses.
    The marching cubes look-up tables are stored in 1D textures.

    6. Render geometry
    Using number of vertices from readback.
*/
// OpenGL Graphics includes
#include <GL/glew.h>
#if defined(__APPLE__) || defined(__MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

//#include <glm/glm.hpp>
//#include <glm/gtc/matrix_transform.hpp>
//#include <glm/gtc/type_ptr.hpp>
//#include <glm/vec3.hpp> // glm::vec3
//#include <glm/vec4.hpp> // glm::vec4
//#include <glm/mat4x4.hpp> // glm::mat4
//#include <glm/ext/matrix_transform.hpp> // glm::translate, glm::rotate, glm::scale
//#include <glm/ext/matrix_clip_space.hpp> // glm::perspective
//#include <glm/ext/scalar_constants.hpp> // glm::pi

#include "camera.hpp"
#include "GLSLShader.h"
#include "MeshProcessing.h"
#pragma comment(lib, "MeshProcessing.lib");
std::string gpuInfo = "Quadro P1000";

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <memory>
#include <iostream>
#include <cassert>

#include <vector>
#include <string>

#include "defines.h"
#include "tables.h"
#include "mc_helper.h"
#include "ScanApple.h"

// standard utility and system includes
#include <oclUtils.h>
#include <shrQATest.h>

// CL/GL includes and defines
#include <CL/cl_gl.h>    

#ifdef UNIX
	#if defined(__APPLE__) || defined(MACOSX)
	    #include <OpenCL/opencl.h>
	    #include <OpenGL/OpenGL.h>
	    #include <GLUT/glut.h>
	    #include <OpenCL/cl_gl_ext.h>
	#else
	    #include <GL/freeglut.h>
	    #include <GL/glx.h>
	#endif
#endif


#if defined (__APPLE__) || defined(MACOSX)
   #define GL_SHARING_EXTENSION "cl_APPLE_gl_sharing"
#else
   #define GL_SHARING_EXTENSION "cl_khr_gl_sharing"
#endif

#define GL_CHECK_ERRORS assert(glGetError()== GL_NO_ERROR);

#define REFRESH_DELAY	  10 //ms

// RAW DATA TYPE
enum RawDataType{
	UCHAR8,	
	USHORT16,
	SHORT16,
	FLOAT32,
};
RawDataType rawType = UCHAR8;

#include "oclScan_common.h"

// OpenCL vars
cl_platform_id cpPlatform;
cl_uint uiNumDevices;
cl_device_id* cdDevices;
cl_uint uiDeviceUsed;
cl_uint uiDevCount;
cl_context cxGPUContext;
cl_device_id device;
cl_command_queue cqCommandQueue;
cl_program cpProgram;
cl_kernel classifyVoxelKernel;
cl_kernel compactVoxelsKernel;
cl_kernel generateTriangles2Kernel;
cl_int ciErrNum;
char* cPathAndName = NULL;          // var for full paths to data, src, etc.
char* cSourceCL;                    // Buffer to hold source for compilation 
cl_bool g_glInterop = false;

int *pArgc = NULL;
char **pArgv = NULL;

class dim3 {
public:
    size_t x;
    size_t y;
    size_t z;

    dim3(size_t _x=1, size_t _y=1, size_t _z=1) {x = _x; y = _y; z = _z;}
};


// constants
const unsigned int window_width = 512;
const unsigned int window_height = 512;

const char *volumeFilename = "Bucky.raw";

#define mc_PI   3.1415926535897932384626433832795
#define mc_2PI  6.283185307179586476925286766559

enum {
	ORTHO = 0,
	PERSPECTIVE = 1,
};
int mc_mode = PERSPECTIVE;
float mc_halfBound[3];
float cameraDistance = 0.0;
float ortho_scale = 1.0;

camera cam;
//glm::mat4 modelMat;
//glm::mat4 mvpMat;

Matrix4 modelMat;
Matrix4 mvpMat;

GLSLShader shader;

cl_uint gridSizeLog2[4] = {5, 5, 5,0};
cl_uint gridSizeShift[4];
cl_uint gridSize[4];
cl_uint gridSizeMask[4];

cl_float mc_scale;
cl_float mc_centerOffset[4];
cl_float UpperLeft[4];


cl_float voxelSize[4];
uint numVoxels    = 0;
uint maxVerts     = 0;
uint activeVoxels = 0;
uint totalVerts   = 0;
uint totalVertsLes = 0;

float isoValue		= 0.0001f;
float dIsoValue		= 0.002f;

float skinColor[] = {1.0,0.5,0.25,0.75};
float lesColor[] = {1.0,0.0,0.0,1.0};

// device data
GLuint posVbo=0, normalVbo=0, pos_normalVbo=0;
GLuint posVao=0;

GLuint posVboLes = 0, normalVboLes = 0, pos_normalVboLes = 0;
GLuint posVaoLes = 0;

GLint  gl_Shader;

cl_mem d_pos = 0;
cl_mem d_normal = 0;
cl_mem d_pos_normal = 0;

cl_mem d_volume = 0;
cl_mem d_voxelVerts = 0;
cl_mem d_voxelVertsScan = 0;
cl_mem d_voxelOccupied = 0;
cl_mem d_voxelOccupiedScan = 0;
cl_mem d_compVoxelArray;

cl_mem d_VertsHash = 0;

cl_mem d_posLes = 0;
cl_mem d_normalLes = 0;
cl_mem d_pos_normalLes = 0;

cl_mem d_volumeLes = 0;
cl_mem d_voxelVertsLes = 0;
cl_mem d_voxelVertsScanLes = 0;
cl_mem d_voxelOccupiedLes = 0;
cl_mem d_voxelOccupiedScanLes = 0;
cl_mem d_compVoxelArrayLes;

cl_mem d_VertsHashLes = 0;

//host data
std::vector<uint> h_VertsHash;
std::vector<float> h_pos;
std::vector<float> h_normal;
std::vector<float> h_pos_normal;


std::vector<uint> h_VertsHashLes;
std::vector<float> h_posLes;
std::vector<float> h_normalLes;
std::vector<float> h_pos_normalLes;

// tables
cl_mem d_numVertsTable = 0;
cl_mem d_triTable = 0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
cl_float mc_rotate[4] = {30.0, -45.0, 0.0, 0.0};
cl_float mc_translate[4] = {0.0, 0.0, 0.0, 0.0};

bool saveMeshFlag = 0;
bool smoothFlag = 1;

// toggles
bool wireframe = false;
bool animate = false;
bool lighting = true;
bool render = true;
bool compute = true;

void Cleanup(int iExitCode);
void (*pCleanup)(int) = &Cleanup;

double totalTime = 0;

// Auto-Verification Code
const int frameCheckNumber = 4;
unsigned int fpsLimit = 100;        // FPS limit for sampling
int g_Index = 0;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_bNoprompt = false;
bool bQATest = false;	
const char* cpExecutableName;

// forward declarations
void runTest(int argc, char** argv);
void initMC(int argc, char** argv);
void computeIsosurface(float isoValue);
void computeIsosurfaceLes(float isoValue);

bool initGL(int argc, char **argv);
void createVBO(GLuint* vbo, unsigned int size, cl_mem &vbo_cl);
void deleteVBO(GLuint* vbo, cl_mem vbo_cl );
//void createVAO(GLuint* vao, unsigned int size, cl_mem &vao_cl);

void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

void idle();
void reshape(int w, int h);
void TestNoGL();

template <class T>
void dumpBuffer(cl_mem d_buffer, T *h_buffer, int nelements);


void mainMenu(int i);

void allocateTextures(	cl_mem *d_triTable, cl_mem* d_numVertsTable )
{
    cl_image_format imageFormat;
    imageFormat.image_channel_order = CL_R;
    imageFormat.image_channel_data_type = CL_UNSIGNED_INT8;
    

    *d_triTable = clCreateImage2D(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  &imageFormat,
                                  16,256,0, (void*) triTable, &ciErrNum );
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    

    *d_numVertsTable = clCreateImage2D(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       &imageFormat,
                                       256,1,0, (void*) numVertsTable, &ciErrNum );
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    
}

//void
//openclScan(cl_mem d_voxelOccupiedScan, cl_mem d_voxelOccupied, int numVoxels) {
//    scanExclusiveLarge(
//                       cqCommandQueue,
//                       d_voxelOccupiedScan,
//                       d_voxelOccupied,
//                       1,
//                       numVoxels);
//    
//    
//}

void
launch_classifyVoxel( dim3 grid, dim3 threads, cl_mem voxelVerts, cl_mem voxelOccupied, cl_mem volume,
					  cl_uint gridSize[4], cl_uint gridSizeShift[4], cl_uint gridSizeMask[4], uint numVoxels,
					  cl_float voxelSize[4], float isoValue)
{
    ciErrNum = clSetKernelArg(classifyVoxelKernel, 0, sizeof(cl_mem), &voxelVerts);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = clSetKernelArg(classifyVoxelKernel, 1, sizeof(cl_mem), &voxelOccupied);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = clSetKernelArg(classifyVoxelKernel, 2, sizeof(cl_mem), &volume);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = clSetKernelArg(classifyVoxelKernel, 3, 4 * sizeof(cl_uint), gridSize);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = clSetKernelArg(classifyVoxelKernel, 4, 4 * sizeof(cl_uint), gridSizeShift);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = clSetKernelArg(classifyVoxelKernel, 5, 4 * sizeof(cl_uint), gridSizeMask);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = clSetKernelArg(classifyVoxelKernel, 6, sizeof(uint), &numVoxels);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = clSetKernelArg(classifyVoxelKernel, 7, 4 * sizeof(cl_float), voxelSize);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = clSetKernelArg(classifyVoxelKernel, 8, sizeof(float), &isoValue);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = clSetKernelArg(classifyVoxelKernel, 9, sizeof(cl_mem), &d_numVertsTable);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    grid.x *= threads.x;
    ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, classifyVoxelKernel, 1, NULL, (size_t*) &grid, (size_t*) &threads, 0, 0, 0);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
}

void
launch_compactVoxels(dim3 grid, dim3 threads, cl_mem compVoxelArray, cl_mem voxelOccupied, cl_mem voxelOccupiedScan, uint numVoxels)
{
    ciErrNum = clSetKernelArg(compactVoxelsKernel, 0, sizeof(cl_mem), &compVoxelArray);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = clSetKernelArg(compactVoxelsKernel, 1, sizeof(cl_mem), &voxelOccupied);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = clSetKernelArg(compactVoxelsKernel, 2, sizeof(cl_mem), &voxelOccupiedScan);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = clSetKernelArg(compactVoxelsKernel, 3, sizeof(cl_uint), &numVoxels);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    
    grid.x *= threads.x;
    ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, compactVoxelsKernel, 1, NULL, (size_t*) &grid, (size_t*) &threads, 0, 0, 0);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
}

void
launch_generateTriangles2(dim3 grid, dim3 threads,
                          cl_mem pos, cl_mem norm, cl_mem pos_norm, cl_mem compactedVoxelArray, cl_mem numVertsScanned, cl_mem volume,
                          cl_uint gridSize[4], cl_uint gridSizeShift[4], cl_uint gridSizeMask[4],
                          cl_float voxelSize[4], cl_float UpperLeft[4], float isoValue, uint activeVoxels, uint maxVerts, cl_mem d_VertsHash)
{
	int k = 0;
    ciErrNum = clSetKernelArg(generateTriangles2Kernel, k++, sizeof(cl_mem), &pos);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = clSetKernelArg(generateTriangles2Kernel, k++, sizeof(cl_mem), &norm);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
	ciErrNum = clSetKernelArg(generateTriangles2Kernel, k++, sizeof(cl_mem), &pos_norm);
	oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = clSetKernelArg(generateTriangles2Kernel, k++, sizeof(cl_mem), &compactedVoxelArray);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = clSetKernelArg(generateTriangles2Kernel, k++, sizeof(cl_mem), &numVertsScanned);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = clSetKernelArg(generateTriangles2Kernel, k++, sizeof(cl_mem), &volume);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS,  pCleanup); 
    ciErrNum = clSetKernelArg(generateTriangles2Kernel, k++, 4 * sizeof(cl_uint), gridSize);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = clSetKernelArg(generateTriangles2Kernel, k++, 4 * sizeof(cl_uint), gridSizeShift);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = clSetKernelArg(generateTriangles2Kernel, k++, 4 * sizeof(cl_uint), gridSizeMask);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = clSetKernelArg(generateTriangles2Kernel, k++, 4 * sizeof(cl_float), voxelSize);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
	ciErrNum = clSetKernelArg(generateTriangles2Kernel, k++, 4 * sizeof(cl_float), UpperLeft);
	oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = clSetKernelArg(generateTriangles2Kernel, k++, sizeof(float), &isoValue);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = clSetKernelArg(generateTriangles2Kernel, k++, sizeof(uint), &activeVoxels);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = clSetKernelArg(generateTriangles2Kernel, k++, sizeof(uint), &maxVerts);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    ciErrNum = clSetKernelArg(generateTriangles2Kernel, k++, sizeof(cl_mem), &d_numVertsTable);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = clSetKernelArg(generateTriangles2Kernel, k++, sizeof(cl_mem), &d_triTable);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
	ciErrNum = clSetKernelArg(generateTriangles2Kernel, k++, sizeof(cl_mem), &d_VertsHash);
	oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    grid.x *= threads.x;
    ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, generateTriangles2Kernel, 1, NULL, (size_t*) &grid, (size_t*) &threads, 0, 0, 0);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

}

void animation()
{
    if (animate) {
        isoValue += dIsoValue;
        if (isoValue < -1.0f ) {
            isoValue = -1.0f;
            dIsoValue *= -1.0f;
        } else if ( isoValue > 1.0f ) {
            isoValue = 1.0f;
            dIsoValue *= -1.0f;
        }
		compute = true;
    }
}

void timerEvent(int value)
{
    animation();
    glutPostRedisplay();
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
}

void computeFPS()
{
    frameCount++;
 
    if (frameCount == fpsLimit) {
        char fps[256];
        float ifps = (float)frameCount / (float)(totalTime);
        sprintf(fps, "OpenCL Marching Cubes: %3.1f fps", ifps);  
        
        glutSetWindowTitle(fps);
        
        frameCount = 0;
        totalTime = 0.0;

        if( g_bNoprompt) Cleanup(EXIT_SUCCESS);        
    }
}


////////////////////////////////////////////////////////////////////////////////
// Load raw data from disk
////////////////////////////////////////////////////////////////////////////////
uchar *loadRawFile(char *filename, int size)
{
	FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error opening file '%s'\n", filename);
        return 0;
    }

	uchar *data = (uchar *) malloc(size);
	size_t read = fread(data, 1, size, fp);
	fclose(fp);

    printf("Read '%s', %d bytes\n", filename, (int)read);

    return data;
}

float* loadRawFilef(const char* filename, size_t size)
{
	FILE *fp = fopen(filename, "rb");
	if (!fp) {
		fprintf(stderr, "Error opening file '%s'\n", filename);
		return 0;
	}

	float *data = (float *)malloc(size);
	size_t read = fread(data, 1, size, fp);
	fclose(fp);

	printf("Read '%s', %d bytes\n", filename, (int)read);

	return data;
}


void initCL(int argc, char** argv) {
    //Get the NVIDIA platform
    ciErrNum = oclGetPlatformID(&cpPlatform);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Get the number of GPU devices available to the platform
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &uiDevCount);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Create the device list
    cdDevices = new cl_device_id [uiDevCount];
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, uiDevCount, cdDevices, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Get device requested on command line, if any
    uiDeviceUsed = 0;
    unsigned int uiEndDev = uiDevCount - 1;
    if(shrGetCmdLineArgumentu(argc, (const char**)argv, "device", &uiDeviceUsed ))
    {
      uiDeviceUsed = CLAMP(uiDeviceUsed, 0, uiEndDev);
      uiEndDev = uiDeviceUsed; 
    } 

	// Check if the requested device (or any of the devices if none requested) supports context sharing with OpenGL
    if(g_glInterop)
    {
        bool bSharingSupported = false;
        for(unsigned int i = uiDeviceUsed; (!bSharingSupported && (i <= uiEndDev)); ++i) 
        {
            size_t extensionSize;
            ciErrNum = clGetDeviceInfo(cdDevices[i], CL_DEVICE_EXTENSIONS, 0, NULL, &extensionSize );
            oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
            if(extensionSize > 0) 
            {
                char* extensions = (char*)malloc(extensionSize);
                ciErrNum = clGetDeviceInfo(cdDevices[i], CL_DEVICE_EXTENSIONS, extensionSize, extensions, &extensionSize);
                oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
                std::string stdDevString(extensions);
                free(extensions);

                size_t szOldPos = 0;
                size_t szSpacePos = stdDevString.find(' ', szOldPos); // extensions string is space delimited
                while (szSpacePos != stdDevString.npos)
                {
                    if( strcmp(GL_SHARING_EXTENSION, stdDevString.substr(szOldPos, szSpacePos - szOldPos).c_str()) == 0 ) 
                    {
                        // Device supports context sharing with OpenGL
                        uiDeviceUsed = i;
                        bSharingSupported = true;
                        break;
                    }
                    do 
                    {
                        szOldPos = szSpacePos + 1;
                        szSpacePos = stdDevString.find(' ', szOldPos);
                    } 
                    while (szSpacePos == szOldPos);
                }
            }
        }
       
        shrLog("%s...\n\n", bSharingSupported ? "Using CL-GL Interop" : "No device found that supports CL/GL context sharing");  
        oclCheckErrorEX(bSharingSupported, true, pCleanup);

        // Define OS-specific context properties and create the OpenCL context
        #if defined (__APPLE__)
            CGLContextObj kCGLContext = CGLGetCurrentContext();
            CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);
            cl_context_properties props[] = 
            {
                CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, (cl_context_properties)kCGLShareGroup, 
                0 
            };
            cxGPUContext = clCreateContext(props, 0,0, NULL, NULL, &ciErrNum);
        #else
            #ifdef UNIX
                cl_context_properties props[] = 
                {
                    CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(), 
                    CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(), 
                    CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform, 
                    0
                };
                cxGPUContext = clCreateContext(props, 1, &cdDevices[uiDeviceUsed], NULL, NULL, &ciErrNum);
            #else // Win32
                cl_context_properties props[] = 
                {
                    CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(), 
                    CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(), 
                    CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform, 
                    0
                };
                cxGPUContext = clCreateContext(props, 1, &cdDevices[uiDeviceUsed], NULL, NULL, &ciErrNum);
            #endif
        #endif
    }
    else 
    {
		// No GL interop
        cl_context_properties props[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform, 0};
        cxGPUContext = clCreateContext(props, 1, &cdDevices[uiDeviceUsed], NULL, NULL, &ciErrNum);

		g_glInterop = false;
    }

    oclPrintDevInfo(LOGBOTH, cdDevices[uiDeviceUsed]);

    // create a command-queue
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevices[uiDeviceUsed], 0, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Program Setup
    size_t program_length;
    cPathAndName = shrFindFilePath("marchingCubes_kernel.cl", argv[0]);
    oclCheckErrorEX(cPathAndName != NULL, shrTRUE, pCleanup);
    cSourceCL = oclLoadProgSource(cPathAndName, "", &program_length);
    oclCheckErrorEX(cSourceCL != NULL, shrTRUE, pCleanup);

    // create the program
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1,
					  (const char **)&cSourceCL, &program_length, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    
    // build the program
    std::string buildOpts = "-cl-mad-enable";
    ciErrNum = clBuildProgram(cpProgram, 0, NULL, buildOpts.c_str(), NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then cleanup and return error
        shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclMarchinCubes.ptx");
        Cleanup(EXIT_FAILURE); 
    }

    // create the kernel
    classifyVoxelKernel = clCreateKernel(cpProgram, "classifyVoxel", &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    compactVoxelsKernel = clCreateKernel(cpProgram, "compactVoxels", &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    generateTriangles2Kernel = clCreateKernel(cpProgram, "generateTriangles2", &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Setup Scan
    //initScan(cxGPUContext, cqCommandQueue, (const char**)argv);
	device = cdDevices[uiDeviceUsed];
	std::string DIR_CL("./");
	int scanflag = MeshProc::scanApple::initScanAPPLE(cxGPUContext, cqCommandQueue, device, DIR_CL);
	if (scanflag < 0) {
		oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
	}

}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{
    pArgc = &argc;
    pArgv = argv;

    shrQAStart(argc, argv);

    cpExecutableName = argv[0];
    shrSetLogFileName ("oclMarchingCubes.txt");
    shrLog("%s Starting...\n\n", argv[0]); 

    if (shrCheckCmdLineFlag(argc, (const char **)argv, "noprompt") ) 
    {
        g_bNoprompt = true;
    }

    if (shrCheckCmdLineFlag(argc, (const char **)argv, "qatest") ) {    
        bQATest = true;
		animate = false;
    }

    runTest(argc, argv);

    Cleanup(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
// initialize marching cubes
////////////////////////////////////////////////////////////////////////////////
void
initMC(int argc, char** argv)
{
    // parse command line arguments
    int n;
    if (shrGetCmdLineArgumenti( argc, (const char**) argv, "grid", &n)) {
        gridSizeLog2[0] = gridSizeLog2[1] = gridSizeLog2[2] = n;
    }
    if (shrGetCmdLineArgumenti( argc, (const char**) argv, "gridx", &n)) {
		gridSize[0] = n;
		//gridSizeLog2[0] = n;
    }
    if (shrGetCmdLineArgumenti( argc, (const char**) argv, "gridy", &n)) {
		gridSize[1] = n;
		//gridSizeLog2[1] = n;
    }
    if (shrGetCmdLineArgumenti( argc, (const char**) argv, "gridz", &n)) {
		gridSize[2] = n;
		//gridSizeLog2[2] = n;
    }
	if (shrGetCmdLineArgumentf(argc, (const char**)argv, "sizex", &voxelSize[0])) {
		//gridSizeLog2[0] = n;
	}
	if (shrGetCmdLineArgumentf(argc, (const char**)argv, "sizey", &voxelSize[1])) {
		//gridSizeLog2[1] = n;
	}
	if (shrGetCmdLineArgumentf(argc, (const char**)argv, "sizez", &voxelSize[2])) {
		//gridSizeLog2[2] = n;
	}

    char *filename;
    if (shrGetCmdLineArgumentstr( argc, (const char**) argv, "file", &filename)) {
        volumeFilename = filename;
    }
	char *maskfilename = NULL;
	if (shrGetCmdLineArgumentstr(argc, (const char**)argv, "mask", &maskfilename)) {
	}
	char *lesfilename = NULL;
	if (shrGetCmdLineArgumentstr(argc, (const char**)argv, "les", &lesfilename)) {
	}


    //gridSize[0] = gridSizeLog2[0];
    //gridSize[1] = gridSizeLog2[1];
    //gridSize[2] = gridSizeLog2[2]+2;
	gridSize[2] += 2;


    gridSizeMask[0] = gridSize[0];
    gridSizeMask[1] = gridSize[1];
    gridSizeMask[2] = gridSize[2];

    gridSizeShift[0] = 1;
    gridSizeShift[1] = gridSize[0];
    gridSizeShift[2] = gridSize[0]*gridSize[1];

    numVoxels = gridSize[0]*gridSize[1]*gridSize[2];

	// compute translate and scale info for MC
	//voxelSize[0] = 0.779297;
	//voxelSize[1] = 0.779297;
	//voxelSize[2] = 4.9444446;
	//voxelSize[0] = 824.21875;
	//voxelSize[1] = 824.21875;
	//voxelSize[2] = 2000.0000000000002;
	float sx = 2.0f / (gridSize[0] * voxelSize[0]);
	float sy = 2.0f / (gridSize[1] * voxelSize[1]);
	if (sx < sy) mc_scale = sx;
	else mc_scale = sy;
	UpperLeft[0] = -169.574997;
	UpperLeft[1] = -52.0999985;
	UpperLeft[2] = 224.460007;
	for (int i = 0; i < 3; ++i) {
		mc_centerOffset[i] = -UpperLeft[i] - voxelSize[i]*(float)(gridSize[i]-1)*0.5;
	}
	mc_halfBound[0] = mc_halfBound[1] = 1.0;
	mc_halfBound[2] = voxelSize[2] * (float)(gridSize[2] - 1)*0.5 * mc_scale;

	cameraDistance = 2.0 * mc_halfBound[0];
	cam.setZplane(mc_halfBound[0] * 0.5, cameraDistance + 2.0*mc_halfBound[0]);

	// set Model Matrix for mc:
	//modelMat = glm::scale(glm::mat4(1.0f), glm::vec3(mc_scale, mc_scale, mc_scale));
	//modelMat = glm::translate(modelMat, glm::vec3(mc_centerOffset[0], mc_centerOffset[1], mc_centerOffset[2]));
	modelMat.identity();
	modelMat.scale(mc_scale);
	modelMat.translate(mc_centerOffset[0], mc_centerOffset[1], mc_centerOffset[2]);

    maxVerts = gridSize[0]*gridSize[1]*36;
    shrLog("grid: %d x %d x %d = %d voxels\n", gridSize[0], gridSize[1], gridSize[2], numVoxels);
    shrLog("max verts = %d\n", maxVerts);

    // load volume data
    char* path = shrFindFilePath(volumeFilename, argv[0]);
    if (path == 0) {
        shrLog("Error finding file '%s'\n", volumeFilename);
        exit(EXIT_FAILURE);
    }
	char* maskpath = shrFindFilePath(maskfilename, argv[0]);
	if (maskpath == 0) {
		shrLog("Error finding file '%s'\n", maskfilename);
		exit(EXIT_FAILURE);
	}
	char* lespath = shrFindFilePath(lesfilename, argv[0]);
	if (lespath == 0) {
		shrLog("Error finding file '%s'\n", lesfilename);
		exit(EXIT_FAILURE);
	}

    int size = gridSize[0]*gridSize[1]*gridSize[2];
	int ori_size = gridSize[0] * gridSize[1] * (gridSize[2] - 2);
	uchar *h_ori_volumeU = NULL;
	uchar *h_ori_maskU = NULL;
	uchar *h_ori_lesU = NULL;
	float *h_ori_volumeF = NULL;
	float* h_volumeF = NULL;
	float* h_ori_lesF = NULL;
	float* h_lesF = NULL;
	h_ori_volumeF = (float*)malloc(ori_size * sizeof(float));
	h_volumeF = (float*)malloc(size * sizeof(float));
	h_ori_lesF = (float*)malloc(ori_size * sizeof(float));
	h_lesF = (float*)malloc(size * sizeof(float));

	switch (rawType) {
	case(UCHAR8): {
		h_ori_volumeU = loadRawFile(path, ori_size);
		h_ori_maskU = loadRawFile(maskpath, ori_size);
		h_ori_lesU = loadRawFile(lespath, ori_size);
		oclCheckErrorEX(h_ori_volumeU != NULL, true, pCleanup);
		shrLog(" Raw file data loaded...\n\n");
		for (size_t i = 0; i < ori_size; ++i) {
			float val = (float)h_ori_volumeU[i] * (float)(h_ori_maskU[i] / 255) / 255.0;
			//float val = (float)(h_ori_volumeU[i]) / 255.0;
			h_ori_volumeF[i] = val;// / 255.0 - 0.5;

			float lesval = (float)h_ori_volumeU[i] * (float)(h_ori_lesU[i] / 255) / 255.0;
			h_ori_lesF[i] = lesval;
		}
		free(h_ori_volumeU);
		free(h_ori_maskU);
		free(h_ori_lesU);
		break;
	}
	case(FLOAT32): {
		h_ori_volumeF = loadRawFilef(path, ori_size * sizeof(float));
		oclCheckErrorEX(h_ori_volumeU != NULL, true, pCleanup);
		shrLog(" Raw file data loaded...\n\n");
	}
	default: {
		break;
	}
	}

	// fill two bound slice for MarchingCuces
	float fmin = h_ori_volumeF[0], fmax = h_ori_volumeF[0];
	for (size_t i = 0; i < ori_size; ++i) {
		if (h_ori_volumeF[i] > fmax) fmax = h_ori_volumeF[i];
		if (h_ori_volumeF[i] < fmin) fmin = h_ori_volumeF[i];
	}

	for (int i = 0; i < gridSize[0] * gridSize[1]; ++i) {
		h_volumeF[i] = fmin;
	}
	for (int i = 0; i < gridSize[0] * gridSize[1]; ++i) {
		h_volumeF[gridSize[0] * gridSize[1] * (gridSize[2] - 1) + i] = fmin;
	}
	memcpy(h_volumeF + gridSize[0] * gridSize[1], h_ori_volumeF, ori_size * sizeof(float));

	fmin = h_ori_lesF[0]; fmax = h_ori_lesF[0];
	for (size_t i = 0; i < ori_size; ++i) {
		if (h_ori_lesF[i] > fmax) fmax = h_ori_lesF[i];
		if (h_ori_lesF[i] < fmin) fmin = h_ori_lesF[i];
	}

	for (int i = 0; i < gridSize[0] * gridSize[1]; ++i) {
		h_lesF[i] = fmin;
	}
	for (int i = 0; i < gridSize[0] * gridSize[1]; ++i) {
		h_lesF[gridSize[0] * gridSize[1] * (gridSize[2] - 1) + i] = fmin;
	}
	memcpy(h_lesF + gridSize[0] * gridSize[1], h_ori_lesF, ori_size * sizeof(float));




	//memcpy(h_volumeF, h_ori_volumeF + gridSize[0] * gridSize[1], gridSize[0] * gridSize[1] * sizeof(float));
	//memcpy(h_volumeF + gridSize[0] * gridSize[1] * (gridSize[2]-1), h_ori_volumeF + gridSize[0] * gridSize[1] * (gridSize[2] - 3), gridSize[0] * gridSize[1] * sizeof(float));

	//float fmin = h_volumeF[0], fmax = h_volumeF[0];
	//for (size_t i = 0; i < size; ++i) {
	//	if (h_volumeF[i] > fmax) fmax = h_volumeF[i];
	//	if (h_volumeF[i] < fmin) fmin = h_volumeF[i];
	//}
	//printf("%f %f \n", fmin, fmax);
	//fmin = -50; fmax = 50;
	//auto bound = [](int x, int l, int h) {if (x < l) return l; if (x > h) return h; return x; };
	//for (size_t i = 0; i < size; ++i) {
	//	float val = h_volumeF[i];
	//	val = (h_volumeF[i] - fmin) / (fmax - fmin) * 255.0;
	//	val = roundf(val);
	//	h_volumeU[i] = (uchar)bound((int)val, 0, 255);
	//}

	// Init OpenCL
    cl_image_format volumeFormat;
    volumeFormat.image_channel_order = CL_R;
    volumeFormat.image_channel_data_type = CL_FLOAT;

    d_volume = clCreateImage3D(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &volumeFormat, 
                                    gridSize[0], gridSize[1], gridSize[2],
                                    gridSize[0]*4, gridSize[0] * gridSize[1]*4,
								h_volumeF, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

	d_volumeLes = clCreateImage3D(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &volumeFormat,
		gridSize[0], gridSize[1], gridSize[2],
		gridSize[0] * 4, gridSize[0] * gridSize[1] * 4,
		h_lesF, &ciErrNum);
	oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

	free(h_ori_volumeF);
	free(h_volumeF);
	free(h_lesF);


    // create VBOs
    if( !bQATest) {
        createVBO(&posVbo, maxVerts*sizeof(float)*4, d_pos);
        createVBO(&normalVbo, maxVerts*sizeof(float)*4, d_normal);
		createVBO(&pos_normalVbo, maxVerts * sizeof(float) * 4 * 2, d_pos_normal);

		createVBO(&posVboLes, maxVerts * sizeof(float) * 4, d_posLes);
		createVBO(&normalVboLes, maxVerts * sizeof(float) * 4, d_normalLes);
		createVBO(&pos_normalVboLes, maxVerts * sizeof(float) * 4 * 2, d_pos_normalLes);
    }

	glutReportErrors();
	// shader:
	GLuint posAttLoc = shader.getAttribute("aPos");
	GLuint normalAttLoc = shader.getAttribute("aNormal");

	glGenVertexArrays(1, &posVao);
	glBindVertexArray(posVao);
	glBindBuffer(GL_ARRAY_BUFFER, posVbo);
	glEnableVertexAttribArray(posAttLoc);
	glVertexAttribPointer(posAttLoc, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(0));
	glBindBuffer(GL_ARRAY_BUFFER, normalVbo);
	glEnableVertexAttribArray(normalAttLoc);
	glVertexAttribPointer(normalAttLoc, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(0));
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
	glutReportErrors();

	glGenVertexArrays(1, &posVaoLes);
	glBindVertexArray(posVaoLes);
	glBindBuffer(GL_ARRAY_BUFFER, posVboLes);
	glEnableVertexAttribArray(posAttLoc);
	glVertexAttribPointer(posAttLoc, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(0));
	glBindBuffer(GL_ARRAY_BUFFER, normalVboLes);
	glEnableVertexAttribArray(normalAttLoc);
	glVertexAttribPointer(normalAttLoc, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(0));
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
	glutReportErrors();

    
    // allocate textures
	allocateTextures(&d_triTable, &d_numVertsTable );

    // allocate device memory
    unsigned int memSize = sizeof(uint) * numVoxels;
    d_voxelVerts = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, memSize, 0, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    d_voxelVertsScan = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, memSize, 0, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    d_voxelOccupied = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, memSize, 0, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    d_voxelOccupiedScan = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, memSize, 0, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    d_compVoxelArray = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, memSize, 0, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
	d_VertsHash = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(uint)*maxVerts, 0, &ciErrNum);
	oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

	d_voxelVertsLes = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, memSize, 0, &ciErrNum);
	oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
	d_voxelVertsScanLes = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, memSize, 0, &ciErrNum);
	oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
	d_voxelOccupiedLes = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, memSize, 0, &ciErrNum);
	oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
	d_voxelOccupiedScanLes = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, memSize, 0, &ciErrNum);
	oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
	d_compVoxelArrayLes = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, memSize, 0, &ciErrNum);
	oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
	d_VertsHashLes = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, sizeof(uint)*maxVerts, 0, &ciErrNum);
	oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
}

void Cleanup(int iExitCode)
{
	//Destroy shader
	shader.DeleteShaderProgram();

    deleteVBO(&posVbo, d_pos);
    deleteVBO(&normalVbo, d_normal);
	deleteVBO(&pos_normalVbo, d_pos_normal);
	glDeleteVertexArrays(1, &posVao);

    if( d_triTable ) clReleaseMemObject(d_triTable);
    if( d_numVertsTable ) clReleaseMemObject(d_numVertsTable);

    if( d_voxelVerts) clReleaseMemObject(d_voxelVerts);
    if( d_voxelVertsScan) clReleaseMemObject(d_voxelVertsScan);
    if( d_voxelOccupied) clReleaseMemObject(d_voxelOccupied);
    if( d_voxelOccupiedScan) clReleaseMemObject(d_voxelOccupiedScan);
    if( d_compVoxelArray) clReleaseMemObject(d_compVoxelArray);

    if( d_volume) clReleaseMemObject(d_volume);
	if (d_VertsHash) clReleaseMemObject(d_VertsHash);

    //closeScan();
	MeshProc::scanApple::closeScanAPPLE();
    
    if(compactVoxelsKernel)clReleaseKernel(compactVoxelsKernel);  
    if(compactVoxelsKernel)clReleaseKernel(generateTriangles2Kernel);  
    if(compactVoxelsKernel)clReleaseKernel(classifyVoxelKernel);  
    if(cpProgram)clReleaseProgram(cpProgram);

    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext)clReleaseContext(cxGPUContext);
    
    // finalize logs and leave
    shrQAFinish2(bQATest, *pArgc, (const char **)pArgv, (iExitCode == 0) ? QA_PASSED : QA_FAILED);

    if ((g_bNoprompt)||(bQATest))
    {
        shrLogEx(LOGBOTH | CLOSELOG, 0, "%s Exiting...\n", cpExecutableName);
    }
    else 
    {
        shrLogEx(LOGBOTH | CLOSELOG, 0, "%s Exiting...\nPress <Enter> to Quit\n", cpExecutableName);
        #ifdef WIN32
            getchar();
        #endif
    }
    exit (iExitCode);
}

void initMenus()
{
    glutCreateMenu(mainMenu);
    glutAddMenuEntry("Toggle animation [ ]", ' ');
    glutAddMenuEntry("Increment isovalue [+]", '+');
    glutAddMenuEntry("Decrement isovalue [-]", '-');
    glutAddMenuEntry("Toggle computation [c]", 'c');
    glutAddMenuEntry("Toggle rendering [r]", 'r');
    glutAddMenuEntry("Toggle lighting [l]", 'l');
    glutAddMenuEntry("Toggle wireframe [w]", 'w');
    glutAddMenuEntry("Quit (esc)", '\033');
    glutAttachMenu(GLUT_RIGHT_BUTTON);
}

void
runTest(int argc, char** argv)
{
    // First initialize OpenGL context, so we can properly set the GL for OpenCL.
    // This is necessary in order to achieve optimal performance with OpenGL/OpenCL interop.
    if( !bQATest ) {
        initGL(argc, argv);
    }
    
    initCL(argc, argv);

	glutReportErrors();
    if( !bQATest ) {
        // register callbacks
        glutDisplayFunc(display);
        glutKeyboardFunc(keyboard);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
        glutIdleFunc(idle);
        glutReshapeFunc(reshape);
        initMenus();
    }
	glutReportErrors();

    // Initialize OpenCL buffers for Marching Cubes 
    initMC(argc, argv);

	glutReportErrors();

    // start rendering mainloop
    if( !bQATest ) {
        glutMainLoop();
    } else {
        TestNoGL();
    }
}

#define DEBUG_BUFFERS 0

////////////////////////////////////////////////////////////////////////////////
//! Run the OpenCL part of the computation
////////////////////////////////////////////////////////////////////////////////
void
computeIsosurface(float isoValue)
{
    int threads = 128;
    dim3 grid(numVoxels / threads, 1, 1);
    // get around maximum grid size of 65535 in each dimension
    //if (grid.x > 65535) {
    //    grid.y = grid.x / 32768;
    //    grid.x = 32768;
    //}

    // calculate number of vertices need per voxel
    launch_classifyVoxel(grid, threads, 
						d_voxelVerts, d_voxelOccupied, d_volume, 
						gridSize, gridSizeShift, gridSizeMask, 
                         numVoxels, voxelSize, isoValue);

    // scan voxel occupied array
	MeshProc::scanApple::ScanAPPLEProcess(d_voxelOccupiedScan, d_voxelOccupied, numVoxels); //openclScan(d_voxelOccupiedScan, d_voxelOccupied, numVoxels);

    // read back values to calculate total number of non-empty voxels
    // since we are using an exclusive scan, the total is the last value of
    // the scan result plus the last value in the input array
    {
        uint lastElement, lastScanElement;

        clEnqueueReadBuffer(cqCommandQueue, d_voxelOccupied,CL_TRUE, (numVoxels-1) * sizeof(uint), sizeof(uint), &lastElement, 0, 0, 0);
        clEnqueueReadBuffer(cqCommandQueue, d_voxelOccupiedScan,CL_TRUE, (numVoxels-1) * sizeof(uint), sizeof(uint), &lastScanElement, 0, 0, 0);

        activeVoxels = lastElement + lastScanElement;
    }

    if (activeVoxels==0) {
        // return if there are no full voxels
        totalVerts = 0;
        return;
    }

    //printf("activeVoxels = %d\n", activeVoxels);

    // compact voxel index array
    launch_compactVoxels(grid, threads, d_compVoxelArray, d_voxelOccupied, d_voxelOccupiedScan, numVoxels);


    // scan voxel vertex count array
	MeshProc::scanApple::ScanAPPLEProcess(d_voxelVertsScan, d_voxelVerts, numVoxels);//openclScan(d_voxelVertsScan, d_voxelVerts, numVoxels);

    // readback total number of vertices
    {
        uint lastElement, lastScanElement;
        clEnqueueReadBuffer(cqCommandQueue, d_voxelVerts,CL_TRUE, (numVoxels-1) * sizeof(uint), sizeof(uint), &lastElement, 0, 0, 0);
        clEnqueueReadBuffer(cqCommandQueue, d_voxelVertsScan,CL_TRUE, (numVoxels-1) * sizeof(uint), sizeof(uint), &lastScanElement, 0, 0, 0);

        totalVerts = lastElement + lastScanElement;
    }

    //printf("totalVerts = %d\n", totalVerts);


    cl_mem interopBuffers[] = {d_pos, d_normal, d_pos_normal};
    
    // generate triangles, writing to vertex buffers
	if( g_glInterop ) {
		// Acquire PBO for OpenCL writing
		glFlush();
		ciErrNum = clEnqueueAcquireGLObjects(cqCommandQueue, 3, interopBuffers, 0, 0, 0);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
    
    dim3 grid2((int) ceil(activeVoxels / (float) NTHREADS), 1, 1);

    //while(grid2.x > 65535) {
    //    grid2.x/=2;
    //    grid2.y*=2;
    //}
    launch_generateTriangles2(grid2, NTHREADS, d_pos, d_normal, d_pos_normal,
                                            d_compVoxelArray, 
                                            d_voxelVertsScan, d_volume, 
                                            gridSize, gridSizeShift, gridSizeMask, 
                                            voxelSize, UpperLeft, isoValue, activeVoxels,
                              maxVerts, d_VertsHash);

	int new_totalVerts = totalVerts;
	h_pos.resize(totalVerts * 4);
	h_normal.resize(totalVerts * 4);
	h_pos_normal.resize(totalVerts * 4 * 2);
	h_VertsHash.resize(totalVerts);
	dumpBuffer(d_pos, h_pos.data(), totalVerts * 4);
	dumpBuffer(d_normal, h_normal.data(), totalVerts * 4);
	dumpBuffer(d_pos_normal, h_pos_normal.data(), totalVerts * 4 * 2);
	dumpBuffer(d_VertsHash, h_VertsHash.data(), totalVerts);
	std::string filename;
	filename = std::string(volumeFilename) + "_" + std::to_string(isoValue) + ".obj";
	if (saveMeshFlag) {
		MC_HELPER::saveMesh(filename, h_pos, h_normal, h_VertsHash);
		saveMeshFlag = 0;
	}

	// mesh filtering:
	MeshProc::MeshData inMesh, inMesh_s, smMesh;
	int nF = totalVerts/ 3;
	int mp_maxF = 1e6;
	bool mp_flag = (nF < mp_maxF);
	if (smoothFlag && !mp_flag) printf("Warning: too much faces, can't do mesh smoothing!!\n");
	std::vector<float> h_pos_sm = h_pos;
	std::vector<float> h_normal_sm = h_normal;
	std::vector<uint> h_VertsHash_sm = h_VertsHash;
	MC_HELPER::getCompactMeshEigen(h_pos, h_VertsHash, h_normal, inMesh.V, inMesh.F, inMesh.N, inMesh.FN);
	if (smoothFlag && mp_flag) {
		std::vector<float> h_pos_sm = h_pos;
		std::vector<float> h_normal_sm = h_normal;
		std::vector<uint> h_VertsHash_sm = h_VertsHash;
		MC_HELPER::getCompactMeshEigen(h_pos, h_VertsHash, h_normal, inMesh.V, inMesh.F, inMesh.N, inMesh.FN);
		//MeshProc::writeMesh("origin.obj", inMesh);

		// clean mesh: remove small regions
		MeshProc::MeshData cleanMesh = inMesh;
		//MeshProc::RemoveSmallRegions(inMesh, cleanMesh, 1e-3);

		//gpuInfostd::string gpuInfo = CCarbonMed3DRecon::getGPUInfo();
		MeshProc::UniformLaplacianSmoothing(cleanMesh, smMesh, 10);
		//MeshProc::CotangentLaplacianSmoothing(cleanMesh, smMesh, 3);
		//MeshProc::BilateralNormalSmoothingGPU(cleanMesh, smMesh, 20, 10, true, gpuInfo);

		MC_HELPER::getArrayFromCompactMesh(h_pos_sm, h_normal_sm, smMesh.V, smMesh.F);
		new_totalVerts = smMesh.F.rows() * 3;
		clEnqueueWriteBuffer(cqCommandQueue, d_pos, CL_TRUE, 0, new_totalVerts * 4 * sizeof(float), h_pos_sm.data(), 0, 0, 0);
		clEnqueueWriteBuffer(cqCommandQueue, d_normal, CL_TRUE, 0, new_totalVerts * 4 * sizeof(float), h_normal_sm.data(), 0, 0, 0);


		// resize
		totalVerts = new_totalVerts;
		uint newSize = totalVerts * 4 * sizeof(float);
		uint copySize = totalVerts * 4 * sizeof(float);

	}




	if( g_glInterop ) {
		// Transfer ownership of buffer back from CL to GL  
		ciErrNum = clEnqueueReleaseGLObjects(cqCommandQueue, 3, interopBuffers, 0, 0, 0);
		oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

		clFinish( cqCommandQueue );
	} 

}

void computeIsosurfaceLes(float isoValue) {
	int threads = 128;
	dim3 grid(numVoxels / threads, 1, 1);
	// get around maximum grid size of 65535 in each dimension
	//if (grid.x > 65535) {
	//    grid.y = grid.x / 32768;
	//    grid.x = 32768;
	//}

	// calculate number of vertices need per voxel
	launch_classifyVoxel(grid, threads,
		d_voxelVertsLes, d_voxelOccupiedLes, d_volumeLes,
		gridSize, gridSizeShift, gridSizeMask,
		numVoxels, voxelSize, isoValue);

	// scan voxel occupied array
	MeshProc::scanApple::ScanAPPLEProcess(d_voxelOccupiedScanLes, d_voxelOccupiedLes, numVoxels); //openclScan(d_voxelOccupiedScan, d_voxelOccupied, numVoxels);

																							// read back values to calculate total number of non-empty voxels
																							// since we are using an exclusive scan, the total is the last value of
																							// the scan result plus the last value in the input array
	{
		uint lastElement, lastScanElement;

		clEnqueueReadBuffer(cqCommandQueue, d_voxelOccupiedLes, CL_TRUE, (numVoxels - 1) * sizeof(uint), sizeof(uint), &lastElement, 0, 0, 0);
		clEnqueueReadBuffer(cqCommandQueue, d_voxelOccupiedScanLes, CL_TRUE, (numVoxels - 1) * sizeof(uint), sizeof(uint), &lastScanElement, 0, 0, 0);

		activeVoxels = lastElement + lastScanElement;
	}

	if (activeVoxels == 0) {
		// return if there are no full voxels
		totalVertsLes = 0;
		return;
	}

	//printf("activeVoxels = %d\n", activeVoxels);

	// compact voxel index array
	launch_compactVoxels(grid, threads, d_compVoxelArrayLes, d_voxelOccupiedLes, d_voxelOccupiedScanLes, numVoxels);


	// scan voxel vertex count array
	MeshProc::scanApple::ScanAPPLEProcess(d_voxelVertsScanLes, d_voxelVertsLes, numVoxels);//openclScan(d_voxelVertsScan, d_voxelVerts, numVoxels);

																					 // readback total number of vertices
	{
		uint lastElement, lastScanElement;
		clEnqueueReadBuffer(cqCommandQueue, d_voxelVertsLes, CL_TRUE, (numVoxels - 1) * sizeof(uint), sizeof(uint), &lastElement, 0, 0, 0);
		clEnqueueReadBuffer(cqCommandQueue, d_voxelVertsScanLes, CL_TRUE, (numVoxels - 1) * sizeof(uint), sizeof(uint), &lastScanElement, 0, 0, 0);

		totalVertsLes = lastElement + lastScanElement;
	}

	//printf("totalVerts = %d\n", totalVerts);


	cl_mem interopBuffers[] = { d_posLes, d_normalLes, d_pos_normalLes };

	// generate triangles, writing to vertex buffers
	if (g_glInterop) {
		// Acquire PBO for OpenCL writing
		glFlush();
		ciErrNum = clEnqueueAcquireGLObjects(cqCommandQueue, 3, interopBuffers, 0, 0, 0);
		oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
	}

	dim3 grid2((int)ceil(activeVoxels / (float)NTHREADS), 1, 1);

	//while(grid2.x > 65535) {
	//    grid2.x/=2;
	//    grid2.y*=2;
	//}
	launch_generateTriangles2(grid2, NTHREADS, d_posLes, d_normalLes, d_pos_normalLes,
		d_compVoxelArrayLes,
		d_voxelVertsScanLes, d_volumeLes,
		gridSize, gridSizeShift, gridSizeMask,
		voxelSize, UpperLeft, isoValue, activeVoxels,
		maxVerts, d_VertsHashLes);

	int new_totalVerts = totalVertsLes;
	h_posLes.resize(totalVertsLes * 4);
	h_normalLes.resize(totalVertsLes * 4);
	h_pos_normalLes.resize(totalVertsLes * 4 * 2);
	h_VertsHashLes.resize(totalVertsLes);
	dumpBuffer(d_posLes, h_posLes.data(), totalVertsLes * 4);
	dumpBuffer(d_normalLes, h_normalLes.data(), totalVertsLes * 4);
	dumpBuffer(d_pos_normalLes, h_pos_normalLes.data(), totalVertsLes * 4 * 2);
	dumpBuffer(d_VertsHashLes, h_VertsHashLes.data(), totalVertsLes);
	std::string filename;
	filename = std::string(volumeFilename) + "_les_" + std::to_string(isoValue) + ".obj";
	if (saveMeshFlag) {
		MC_HELPER::saveMesh(filename, h_posLes, h_normalLes, h_VertsHashLes);
		saveMeshFlag = 0;
	}

	// mesh filtering:
	MeshProc::MeshData inMesh, inMesh_s, smMesh;
	int nF = totalVertsLes / 3;
	int mp_maxF = 1e6;
	bool mp_flag = (nF < mp_maxF);
	if (smoothFlag && !mp_flag) printf("Warning: too much faces, can't do mesh smoothing!!\n");
	MC_HELPER::getCompactMeshEigen(h_posLes, h_VertsHashLes, h_normalLes, inMesh.V, inMesh.F, inMesh.N, inMesh.FN);
	if (smoothFlag && mp_flag) {
		std::vector<float> h_pos_sm = h_posLes;
		std::vector<float> h_normal_sm = h_normalLes;
		std::vector<uint> h_VertsHash_sm = h_VertsHashLes;
		MC_HELPER::getCompactMeshEigen(h_posLes, h_VertsHashLes, h_normalLes, inMesh.V, inMesh.F, inMesh.N, inMesh.FN);
		//MeshProc::writeMesh("origin.obj", inMesh);

		// clean mesh: remove small regions
		MeshProc::MeshData cleanMesh = inMesh;
		//MeshProc::RemoveSmallRegions(inMesh, cleanMesh, 1e-3);

		//gpuInfostd::string gpuInfo = CCarbonMed3DRecon::getGPUInfo();
		MeshProc::UniformLaplacianSmoothing(cleanMesh, smMesh, 10);
		//MeshProc::CotangentLaplacianSmoothing(cleanMesh, smMesh, 3);
		//MeshProc::BilateralNormalSmoothingGPU(cleanMesh, smMesh, 20, 10, true, gpuInfo);

		MC_HELPER::getArrayFromCompactMesh(h_pos_sm, h_normal_sm, smMesh.V, smMesh.F);
		new_totalVerts = smMesh.F.rows() * 3;
		clEnqueueWriteBuffer(cqCommandQueue, d_posLes, CL_TRUE, 0, new_totalVerts * 4 * sizeof(float), h_pos_sm.data(), 0, 0, 0);
		clEnqueueWriteBuffer(cqCommandQueue, d_normalLes, CL_TRUE, 0, new_totalVerts * 4 * sizeof(float), h_normal_sm.data(), 0, 0, 0);


		// resize
		totalVertsLes = new_totalVerts;
		uint newSize = totalVertsLes * 4 * sizeof(float);
		uint copySize = totalVertsLes * 4 * sizeof(float);

	}




	if (g_glInterop) {
		// Transfer ownership of buffer back from CL to GL  
		ciErrNum = clEnqueueReleaseGLObjects(cqCommandQueue, 3, interopBuffers, 0, 0, 0);
		oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

		clFinish(cqCommandQueue);
	}
}



//// shader for displaying floating-point texture
//static const char *shader_code = 
//"!!ARBfp1.0\n"
//"TEX result.color, fragment.texcoord, texture[0], 2D; \n"
//"END";
//
//GLuint compileASMShader(GLenum program_type, const char *code)
//{
//    GLuint program_id;
//    glGenProgramsARB(1, &program_id);
//    glBindProgramARB(program_type, program_id);
//    glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei) strlen(code), (GLubyte *) code);
//
//    GLint error_pos;
//    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);
//    if ((int)error_pos != -1) {
//        const GLubyte *error_string;
//        error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
//        shrLog("Program error at position: %d\n%s\n", (int)error_pos, error_string);
//        return 0;
//    }
//    return program_id;
//}

////////////////////////////////////////////////////////////////////////////////
//! Initialize OpenGL
////////////////////////////////////////////////////////////////////////////////
bool
initGL(int argc, char **argv)
{
    // Create GL context
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitContextVersion(3, 3);
	glutInitContextFlags(GLUT_COMPATIBILITY_PROFILE | GLUT_DEBUG);
	glutInitContextProfile(GLUT_FORWARD_COMPATIBLE);
	glutInitWindowSize(window_width, window_height);
    glutCreateWindow("CUDA Marching Cubes");
#if !(defined (__APPLE__) || defined(MACOSX))
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
#endif
	
    // initialize necessary OpenGL extensions
	GLenum err = glewInit();
	if (GLEW_OK != err) {
		cerr << "Error: " << glewGetErrorString(err) << endl;
	}
	else {
		if (GLEW_VERSION_3_3)
		{
			cout << "Driver supports OpenGL 3.3\nDetails:" << endl;
		}
	}
    //if (! glewIsSupported("GL_VERSION_2_0 " 
		  //                )) {
    //    fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
    //    fflush(stderr);
    //    return false;
    //}

    // default initialization
	//glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
	glEnable(GL_DEPTH_TEST);

	// good old - fashioned fixed function lighting
	float black[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	float white[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	float ambient[] = { 0.1f, 0.1f, 0.1f, 1.0f };
	float diffuse[] = { 0.9f, 0.9f, 0.9f, 1.0f };
	float lightPos[] = { 0.0f, 0.0f, 1.0f, 0.0f };

	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambient);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, black);

	glLightfv(GL_LIGHT0, GL_AMBIENT, white);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, white);
	glLightfv(GL_LIGHT0, GL_SPECULAR, white);
	glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, black);

	glEnable(GL_LIGHT0);
	glEnable(GL_NORMALIZE);
    
    // load shader program
    //gl_Shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);

	GL_CHECK_ERRORS
	//load shader
	//shader.LoadFromFile(GL_VERTEX_SHADER, "shader.vert");
	//shader.LoadFromFile(GL_FRAGMENT_SHADER, "shader.frag");
	shader.LoadFromFile(GL_VERTEX_SHADER, "volMesh.vert");
	shader.LoadFromFile(GL_FRAGMENT_SHADER, "volMesh.frag");
	//compile and link shader
	shader.CreateAndLinkProgram();
	shader.Use();
	//add shader attribute and uniforms
	//shader.AddAttribute("vVertex");
	//shader.AddUniform("MVP");
	//shader.AddAttribute("aPos");
	//shader.AddAttribute("aNormal");
	//shader.AddUniform("model");
	//shader.AddUniform("view");
	//shader.AddUniform("projection");
	//shader.AddUniform("lightPos");
	//shader.AddUniform("viewPos");
	//shader.AddUniform("lightColor");
	//shader.AddUniform("objectColor");




	shader.UnUse();


	GL_CHECK_ERRORS

	glutReportErrors();

    g_glInterop = true;

    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void
createVBO(GLuint* vbo, unsigned int size, cl_mem &vbo_cl)
{
    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glutReportErrors();

    vbo_cl = clCreateFromGLBuffer(cxGPUContext,CL_MEM_WRITE_ONLY, *vbo, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void
deleteVBO(GLuint* vbo, cl_mem vbo_cl)
{
    if( vbo_cl) clReleaseMemObject(vbo_cl);    

    if( *vbo ) {
        glBindBuffer(1, *vbo);
        glDeleteBuffers(1, vbo);
        
        *vbo = 0;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Render isosurface geometry from the vertex buffers
////////////////////////////////////////////////////////////////////////////////
void renderIsosurface()
{
    glBindBuffer(GL_ARRAY_BUFFER, posVbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);

    glBindBufferARB(GL_ARRAY_BUFFER_ARB, normalVbo);
    glNormalPointer(GL_FLOAT, sizeof(float)*4, 0);
    glEnableClientState(GL_NORMAL_ARRAY);

	glEnable(GL_COLOR_MATERIAL);
    //glColor3f(0.0, 0.7, 0.4);
	glColor4f(0.4, 0.0, 0.0, 0.5);
    glDrawArrays(GL_TRIANGLES, 0, totalVerts);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

////////////////////////////////////////////////////////////////////////////////
// Render isosurface geometry from the vertex buffers using shaders!
////////////////////////////////////////////////////////////////////////////////
void SetLightsForRendering_shader() {
	float lightPos0[] = { 0.0f, 0.0f, -1.0f, 0.0f };
	float lightPos1[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	float lightPos2[] = { 1.0f, 0.0f, 0.0f, 0.0f };
	float lightPos3[] = { -1.0f, 0.0f, 0.0f, 0.0f };
	float lightPos4[] = { 0.0, -1.0, 0.0, 0.0f };
	float lightPos5[] = { 0.0, 1.0, 0.0, 0.0f };

	float lightblack[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	float lightwhite[] = { 0.8f, 0.8f, 0.8f, 1.0f };
	float lightambient[] = { 0.1f, 0.1f, 0.1f, 1.0f };
	float lightdiffuse[] = { 0.3f, 0.3f, 0.3f, 1.0f };
	float lightspecular[] = { 0.3f, 0.3f, 0.3f, 1.0f };

	int lightNum = 2;
	//glUniform1ui(m_volShader("lightNum"), lightNum);
	shader.setInt("lightNum", lightNum);
	// light 0
	//glUniform4fv(m_volShader("lights[0].position"), 1, lightPos0);
	//glUniform4fv(m_volShader("lights[0].ambient"), 1, lightwhite);
	//glUniform4fv(m_volShader("lights[0].diffuse"), 1, lightwhite);
	//glUniform4fv(m_volShader("lights[0].specular"), 1, lightspecular);
	shader.setVec4("lights[0].position", lightPos0);
	shader.setVec4("lights[0].ambient", lightwhite);
	shader.setVec4("lights[0].diffuse", lightwhite);
	shader.setVec4("lights[0].specular", lightspecular);
	// light 1
	//glUniform4fv(m_volShader("lights[1].position"), 1, lightPos1);
	//glUniform4fv(m_volShader("lights[1].ambient"), 1, lightwhite);
	//glUniform4fv(m_volShader("lights[1].diffuse"), 1, lightwhite);
	//glUniform4fv(m_volShader("lights[1].specular"), 1, lightspecular);
	shader.setVec4("lights[1].position", lightPos1);
	shader.setVec4("lights[1].ambient", lightwhite);
	shader.setVec4("lights[1].diffuse", lightwhite);
	shader.setVec4("lights[1].specular", lightspecular);
}

void SetRenderFeatureSkin_shader() {
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);
	//glEnable(GL_DEPTH_TEST);
	glDisable(GL_DEPTH_TEST);
	int wireframe = false;
	glPolygonMode(GL_FRONT, wireframe ? GL_LINE : GL_FILL);

	float black[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	float white[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	float ambient[] = { 0.1f, 0.1f, 0.1f, 1.0f };
	float diffuse[] = { 0.8f, 0.8f, 0.8f, 1.0f };
	Vector4 lightPos(0.0f, 0.0f, -1.0f, 0.0f);

	Matrix4 mv = cam.getViewMat4() * modelMat;
	lightPos = mv * lightPos;

	shader.setVec4("material.ambient", ambient);
	shader.setVec4("material.diffuse", diffuse);
	shader.setVec4("material.specular", black);
	shader.setFloat("material.shiness", 5.0f);

	shader.setVec4("lights[0].position", lightPos.get());
	shader.setVec4("lights[0].ambient", black);
	shader.setVec4("lights[0].diffuse", diffuse);
	shader.setVec4("lights[0].specular", white);

	float paintColor[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	shader.setVec4("paintColor", skinColor);
}

void SetRenderFeatureOrgan_shader() {
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_DEPTH_TEST);
	int wireframe = false;
	glPolygonMode(GL_FRONT, wireframe ? GL_LINE : GL_FILL);

	float black[] = { 0.5f, 0.5f, 0.5f, 1.0f };
	GLfloat afAmbientWhite[] = { 0.25, 0.25, 0.25, 1.00 };   // ÖÜÎ§ »·ÈÆ °× 
	GLfloat afAmbientRed[] = { 0.25, 0.00, 0.00, 1.00 };     // ÖÜÎ§ »·ÈÆ ºì
	GLfloat afDiffuseWhite[] = { 0.75, 0.75, 0.75, 1.00 };   // ÂþÉä °×
	GLfloat afDiffuseRed[] = { 0.75, 0.00, 0.00, 0.00 };     // ÂþÉä ºì
	GLfloat afSpecularRed[] = { 1.00, 0.25, 0.25, 1.00 };    // ·´Éä ºì

	shader.setVec4("material.ambient", afAmbientRed);
	shader.setVec4("material.diffuse", lesColor);
	shader.setVec4("material.specular", black);
	shader.setFloat("material.shiness", 5.0f);

	float paintColor[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	shader.setVec4("paintColor", paintColor);
}

void renderIsosurface_shader()
{
	//bind the shader
	shader.Use();
	SetLightsForRendering_shader();
	
	//SetRenderFeatureOrgan_shader();
	glPolygonMode(GL_FRONT, wireframe? GL_LINE:GL_FILL);

	//clear the colour and depth buffer
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	//mvpMat = cam.getProjMatrixData() * cam.getViewMatrixData() * modelMat;
	mvpMat = cam.getProjMat4() * cam.getViewMat4() * modelMat;
	//pass the shader uniform
	//glUniformMatrix4fv(shader("MVP"), 1, GL_FALSE, glm::value_ptr(mvpMat));
	//glUniformMatrix4fv(shader("projection"), 1, GL_FALSE, cam.getProjMatrixDataTransPosePtr());
	//glUniformMatrix4fv(shader("view"), 1, GL_FALSE, cam.getViewMatrixDataTransPosePtr());
	//glUniformMatrix4fv(shader("model"), 1, GL_FALSE, modelMat.getTranspose());
	shader.setMat4("projection", cam.getProjMatrixDataTransPosePtr());
	shader.setMat4("view", cam.getViewMatrixDataTransPosePtr());
	shader.setMat4("model", modelMat.getTranspose());
	//glm::vec3 lightPos(0.0, 0.0, 2.0);
	//glUniform3fv(shader("lightPos"), 1, glm::value_ptr(lightPos));
	//glm::vec3 viewPos(0.0, 0.0, 0.0);
	//glUniform3fv(shader("viewPos"), 1, glm::value_ptr(viewPos));
	//glm::vec3 lightColor(1.0, 1.0, 1.0);
	//glUniform3fv(shader("lightColor"), 1, glm::value_ptr(lightColor));
	//glm::vec3 objectColor(1.0, 0.0, 0.0);
	//glUniform3fv(shader("objectColor"), 1, glm::value_ptr(objectColor));
	//draw triangle


	SetRenderFeatureOrgan_shader();
	glBindVertexArray(posVaoLes); 
	glDrawArrays(GL_TRIANGLES, 0, totalVertsLes);
	glBindVertexArray(0);

	SetRenderFeatureSkin_shader();
	glBindVertexArray(posVao);
	glDrawArrays(GL_TRIANGLES, 0, totalVerts);
	glBindVertexArray(0);

	//unbind the shader
	shader.UnUse();



	//swap front and back buffers to show the rendered result
	//glutSwapBuffers();

}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void
display()
{
    shrDeltaT(0);

	glutReportErrors();
    // run CUDA kernel to generate geometry
    if (compute) {
        computeIsosurface(isoValue);
		computeIsosurfaceLes(isoValue);
		compute = false;
    }
	glutReportErrors();
	glEnable(GL_NORMALIZE);
    // Common display code path
	{
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// set view matrix
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glTranslatef(0, 0, -cameraDistance);
		glTranslatef(mc_translate[0], mc_translate[1], mc_translate[2]);
		glRotatef(mc_rotate[0], 1.0, 0.0, 0.0);
		glRotatef(mc_rotate[1], 0.0, 1.0, 0.0);
		glRotatef(180.0, 0.0, 1.0, 0.0); // heading

		glPolygonMode(GL_FRONT, wireframe? GL_LINE : GL_FILL);
		if (lighting) {
			glEnable(GL_LIGHTING);
		}

		// render
		if (render) {
			glPushMatrix();
			//glLoadIdentity();
			//glMultMatrixf(cam.getViewMatrixDataPtr());
			//glRotatef(180.0, 0.0, 1.0, 0.0);
			//glRotatef(-90.0, 1.0, 0.0, 0.0);
			if (mc_mode == ORTHO) {
				glScalef(ortho_scale, ortho_scale, ortho_scale);
			}
			glScalef(mc_scale, mc_scale, mc_scale);
			glTranslatef(mc_centerOffset[0], mc_centerOffset[1], mc_centerOffset[2]);

			////float mat[16];
			////glGetFloatv(GL_MODELVIEW_MATRIX, mat);
			////for (int i = 0; i < 16; ++i) printf("%f ", mat[i]);
			////for (int i = 0; i < 16; ++i) printf("%f ", glm::value_ptr(modelMat)[i]);

			//renderIsosurface();
			////renderIsosurface_shader();

			glPopMatrix();

			glLoadIdentity();
			//glMultMatrixf(cam.getProjMatrixDataPtr());
			glMultMatrixf(cam.getViewMatrixDataTransPosePtr());
			glMultMatrixf(modelMat.getTranspose());
			//renderIsosurface();

			renderIsosurface_shader();
			
		}

		//glDisable(GL_LIGHTING);
	} 

    totalTime += shrDeltaT(0);

    glutSwapBuffers();
    glutReportErrors();

    computeFPS();
}


////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void
keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch(key) {
    case '\033':// Escape quits    
    case 'Q':   // Q quits
    case 'q':   // q quits
        g_bNoprompt = true;
		Cleanup(EXIT_SUCCESS);
		break;
    case '=':
        isoValue += 0.01f;
        break;
    case '-':
        isoValue -= 0.01f;
        break;
    case '+':
        isoValue += 0.1f;
        break;
    case '_':
        isoValue -= 0.1f;
        break;
    case 'w':
        wireframe = !wireframe;
        break;
    case ' ':
        animate = !animate;
        break;
    case 'l':
        lighting = !lighting;
        break;
    case 'r':
        render = !render;
        break;
    case 'c':
        compute = !compute;
        break;
	case 'p':
		mc_mode = 1 - mc_mode;
		ortho_scale = 1.0;
		for (int i = 0; i < 4; ++i) mc_translate[i] = 0.0;
		mc_rotate[0] = 30.0;  mc_rotate[1] = -45.0; mc_rotate[2] = 0.0; mc_rotate[3] = 0.0;
		mc_translate[0] = mc_translate[1] = mc_translate[2] = mc_translate[3] = 0.0;
		cam.changeProjMode();
		glutReshapeWindow(window_width, window_height);
		break;
	case 's':
		saveMeshFlag = 1;
		break;
	default:
		break;
    }

    printf("isoValue = %f\n", isoValue);
    printf("voxels = %d\n", activeVoxels);
    printf("verts = %d\n", totalVerts);
    printf("occupancy: %d / %d = %.2f%%\n", 
           activeVoxels, numVoxels, activeVoxels*100.0f / (float) numVoxels);

	if (!compute) {
		compute = 1;
		//computeIsosurface(isoValue);
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void
mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN) {
        mouse_buttons |= 1<<button;
    } else if (state == GLUT_UP) {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    float dx = (float)(x - mouse_old_x);
    float dy = (float)(y - mouse_old_y);

    if (mouse_buttons==1) {
        mc_rotate[0] += dy * 0.2f;
        mc_rotate[1] += dx * 0.2f;
		if (mc_rotate[0] < 0.0)   mc_rotate[0] += 360.0;
		if (mc_rotate[0] > 360.0) mc_rotate[0] -= 360.0;
		if (mc_rotate[1] < 0.0)   mc_rotate[1] += 360.0;
		if (mc_rotate[1] > 360.0) mc_rotate[1] -= 360.0;

		//cam.setRotate(mc_rotate[0], mc_rotate[1], 0.0);
		cam.addRotate(dy, dx, 0.0f);
    } 
	else if (mouse_buttons==2) {
        mc_translate[0] -= dx * 0.005f;
        mc_translate[1] += dy * 0.005f;
		if (mc_translate[0] > 1.5)   mc_translate[0] = 1.5;
		if (mc_translate[0] < -1.5)  mc_translate[0] = -1.5;
		if (mc_translate[1] > 1.5)   mc_translate[1] = 1.5;
		if (mc_translate[1] < -1.5)  mc_translate[1] = -1.5;

		//cam.setCamPos(-dx, dy, 0.0);
		cam.addCamPos(-dx, dy, 0.0);
    } 
	else if (mouse_buttons==3) {
		if (mc_mode == ORTHO) {
			ortho_scale += dy * 0.01;
			if (ortho_scale > 8.0) ortho_scale = 8.0;
			if (ortho_scale < 0.1) ortho_scale = 0.1;
		}
		else if (mc_mode == PERSPECTIVE) {
			mc_translate[2] += dy * 0.005f;
			if (mc_translate[2] > cameraDistance)   mc_translate[2] = cameraDistance;
			if (mc_translate[2] < -3.0*mc_halfBound[0])  mc_translate[2] = -3.0*mc_halfBound[0];
		}
		cam.addScale(dy);
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void idle()
{
    animation();
    glutPostRedisplay();
}

void reshape(int w, int h)
{
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	if (mc_mode == ORTHO)
	{
		if (w > h)
			glOrtho(-2.0*float(w) / float(h), 2.0*float(w) / float(h), -2.0, 2.0, -(-cameraDistance + 1000), -(-cameraDistance - 1000));
			//glOrtho(-2.0*float(w) / float(h), 2.0*float(w) / float(h), -2.0, 2.0, -(-mc_halfBound[0] * 0.5), -(-cameraDistance - 2.0*mc_halfBound[0]));
		else
			glOrtho(-2.0, 2.0, -2.0*float(h) / float(w), 2.0*float(h) / float(w), -(-cameraDistance + 1000), -(-cameraDistance - 1000));
			//glOrtho(-2.0, 2.0, -2.0*float(h) / float(w), 2.0*float(h) / float(w), -(-mc_halfBound[0] * 0.5), -(-cameraDistance - 2.0*mc_halfBound[0]));
	}
	else if (mc_mode == PERSPECTIVE)
	{
		if (w > h)
			gluPerspective(60.0f, float(w) / float(h), -(-mc_halfBound[0]*0.5), -(-cameraDistance - 2.0*mc_halfBound[0]));
		else
			gluPerspective(2.0*atan(float(h)/float(w)*tan(0.5f*60.0f*mc_PI/180.0f))*180.0 / mc_PI, float(w) / float(h), -(-mc_halfBound[0]*0.5), -(-cameraDistance - 2.0*mc_halfBound[0]));
	}
	//glGetFloatv(GL_PROJECTION_MATRIX, cam.getProjMatrixDataPtr());

	//cam.setZplane(mc_halfBound[0] * 0.5, cameraDistance + 2.0*mc_halfBound[0]);
	cam.setWindowSize(w, h);
	glLoadMatrixf(cam.getProjMatrixDataTransPosePtr());


    //glMatrixMode(GL_PROJECTION);
    //glLoadIdentity();
    //gluPerspective(60.0, (float) w / (float) h, 0.1, 10.0);

    glMatrixMode(GL_MODELVIEW);
    //glViewport(0, 0, w, h);
}

void mainMenu(int i)
{
    keyboard((unsigned char) i, 0, 0);
}

template <class T>
void dumpBuffer(cl_mem d_buffer, T *h_buffer, int nelements) {
    clEnqueueReadBuffer(cqCommandQueue,d_buffer, CL_TRUE, 0,nelements * sizeof(T), h_buffer, 0, 0, 0);
}

// Run a test sequence without any GL 
//*****************************************************************************
void TestNoGL()
{
    d_normal = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY,  maxVerts*sizeof(float)*4, NULL, &ciErrNum);
    d_pos = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY,  maxVerts*sizeof(float)*4, NULL, &ciErrNum);
        
    
    // Warmup
    computeIsosurface(isoValue);
    clFinish(cqCommandQueue);
    
    // Start timer 0 and process n loops on the GPU 
    shrDeltaT(0); 
    int nIter = 100;

    for (int i = 0; i < nIter; i++)
    {
        computeIsosurface(isoValue);
    }
    clFinish(cqCommandQueue);
    
    // Get elapsed time and throughput, then log to sample and master logs
    double dAvgTime = shrDeltaT(0)/nIter;
    shrLogEx(LOGBOTH | MASTER, 0, "oclMarchingCubes, Throughput = %.4f MVoxels/s, Time = %.5f s, Size = %u Voxels, NumDevsUsed = %u, Workgroup = %u\n", 
           (1.0e-6 * numVoxels)/dAvgTime, dAvgTime, numVoxels, 1, NTHREADS); 
}