#pragma once
#include <Eigen/Eigen>
#include <vector>
#include <string>
#include <stdio.h>

typedef unsigned int uint;


namespace MC_HELPER {
	void saveMesh(std::string filename, std::vector<float> &verts, std::vector<float> &fNormals, std::vector<uint> &vHashes);

	void getCompactMeshEigen(std::vector<float> &verts, std::vector<uint> &vHashes, std::vector<float> &fNormals,
							Eigen::MatrixXf &V, Eigen::MatrixXi &F, Eigen::MatrixXf &vN, Eigen::MatrixXi &FN);
	void getOriginMeshEigen(std::vector<float> &verts, Eigen::MatrixXf &V, Eigen::MatrixXi &F);
	void getArrayFromCompactMesh(std::vector<float> &verts, std::vector<float> &normals, Eigen::MatrixXf &V, Eigen::MatrixXi &F);

};