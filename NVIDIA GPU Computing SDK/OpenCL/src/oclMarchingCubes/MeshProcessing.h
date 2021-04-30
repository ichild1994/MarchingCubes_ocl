#pragma once
#include <Eigen/Eigen>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <vector>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <memory>
#include <iostream>
#include <cassert>


namespace MeshProc {
	typedef struct RawMatrixMesh {
		Eigen::MatrixXf V, N, TC, FTC;
		Eigen::MatrixXi F, FN;
		void clear() {}
	}MeshData;

	/////IO
	__declspec(dllexport) bool readMesh(std::string fn, MeshData &mesh);
	__declspec(dllexport) bool writeMesh(std::string fn, MeshData &mesh);
	__declspec(dllexport) bool readOBJ(std::string fn, MeshData &mesh);
	__declspec(dllexport) bool writeOBJ(std::string fn, MeshData &mesh);

	////Mesh Smoothing
	__declspec(dllexport) void BilateralNormalSmoothing(MeshData &inMesh, MeshData &outMesh, int itersN = 20, int itersV = 10, bool preserveBound = true);
	//void BilateralNormalFilteringGPU(MeshData &inMesh, MeshData &outMesh);
	__declspec(dllexport) void CotangentLaplacianSmoothing(MeshData &inMesh, MeshData &outMesh, int iters = 3, bool preserveBound = true);
	//void CotangentLaplacianSmoothingGPU(MeshData &inMesh, MeshData &outMesh, int iters = 3, bool preserveBound = true);
	__declspec(dllexport) void UniformLaplacianSmoothing(MeshData &inMesh, MeshData &outMesh, int iters = 3, bool preserveBound = true);

	__declspec(dllexport) void BilateralNormalSmoothingGPU(MeshData &inMesh, MeshData &outMesh, int itersN = 20, int itersV = 10, bool preserveBound = true, std::string gpuInfo = "Quadro P1000", std::string DIR_CL = "D:/CarbonMed/Config/Local/CL/");
	__declspec(dllexport) void GridSimplification(MeshData &inMesh, MeshData &outMesh, int maxDepth = 7);

	////Mesh Processing Utilities
	__declspec(dllexport) void RemoveSmallRegions(MeshData &inMesh, MeshData &outMesh, float ratio = 1e-3);

};
