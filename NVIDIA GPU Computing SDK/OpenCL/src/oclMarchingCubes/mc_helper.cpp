#include "mc_helper.h"
#include <map>
#include <igl/writeOBJ.h>

namespace MC_HELPER {
	void saveMesh(std::string filename, std::vector<float> &verts, std::vector<float> &fNormals, std::vector<uint> &vHashes) {


		//void getCompactMesh(std::vector<float> &verts,std::vector<float> &fNormals,std::vector<uint> &vHashes,
		//					std::vector<float> &compactVerts, std::vector<int> &faces);
		Eigen::MatrixXf vV, vN, TC, FTC;
		Eigen::MatrixXi F, FN;
		getCompactMeshEigen(verts, vHashes, fNormals, vV, F, vN, FN);
		//getOriginMeshEigen(verts, vV, F);
		//igl::writeOBJ(filename, vV, F, vN, FN, TC, FTC);
		bool flag = igl::writeOBJ(filename, vV, F);
		if (flag) {
			printf("save %s succeed!\n", filename.c_str());
		}
		else {
			printf("save %s failed!\n", filename.c_str());
		}
	}

	void getCompactMeshEigen(std::vector<float> &verts, std::vector<uint> &vHashes, std::vector<float> &fNormals, 
		Eigen::MatrixXf &V, Eigen::MatrixXi &F, Eigen::MatrixXf &vN, Eigen::MatrixXi &FN) {

		std::map<uint, int> vHashMap;
		std::vector<uint> revMap;
		uint cnt = 0;
		for (uint i = 0; i < vHashes.size(); ++i) {
			uint id = vHashes[i];
			if (vHashMap.find(id) == vHashMap.end()) {
				//V(cnt, 0) = verts[i * 4];
				//V(cnt, 1) = verts[i * 4 + 1];
				//V(cnt, 2) = verts[i * 4 + 2];
				revMap.emplace_back(i);
				vHashMap[id] = cnt++;
			}
		}
		V.resize(cnt, 3);
		for (uint i = 0; i < cnt; ++i) {
			uint id = revMap[i];
			V(i, 0) = verts[id * 4];
			V(i, 1) = verts[id * 4 + 1];
			V(i, 2) = verts[id * 4 + 2];
		}

		//std::cout << V << std::endl;


		uint numFaces = verts.size() / 12;
		vN.resize(numFaces, 3);
		F.resize(numFaces, 3);
		FN.resize(numFaces, 3);
		for (uint i = 0; i < F.rows(); ++i) {
			F(i, 0) = vHashMap[vHashes[i*3]];
			F(i, 1) = vHashMap[vHashes[i*3+1]];
			F(i, 2) = vHashMap[vHashes[i*3+2]];
			vN(i, 0) = fNormals[i * 12];
			vN(i, 1) = fNormals[i * 12 + 1];
			vN(i, 2) = fNormals[i * 12 + 2];
			FN(i, 0) = i;
			FN(i, 1) = i;
			FN(i, 2) = i;
		}

	}

	void getArrayFromCompactMesh(std::vector<float> &verts, std::vector<float> &normals, Eigen::MatrixXf &V, Eigen::MatrixXi &F) {
		int nV = V.rows();
		int nF = F.rows();
		verts.resize(nF * 3 * 4);
		normals.resize(nF * 3 * 4);
		int vp = 0, fp = 0;
		for (int i = 0; i < nF; ++i) {
			Eigen::Vector3f pt[3];
			for (int j = 0; j < 3; ++j) {
				int vid = F(i, j);
				pt[j] = V.row(vid);
				verts[vp++] = pt[j][0];
				verts[vp++] = pt[j][1];
				verts[vp++] = pt[j][2];
				verts[vp++] = 1.0;
			}
			Eigen::Vector3f normal = (pt[2] - pt[0]).cross(pt[1] - pt[0]);
			normal.normalize();
			for (int j = 0; j < 3; ++j) {
				normals[fp++] = normal[0];
				normals[fp++] = normal[1];
				normals[fp++] = normal[2];
				normals[fp++] = 1.0;
			}
		}
	}


	void getOriginMeshEigen(std::vector<float> &verts, Eigen::MatrixXf &V, Eigen::MatrixXi &F) {
		uint numV = verts.size() / 4;
		uint numF = numV / 3;
		V.resize(numV, 3);
		F.resize(numF, 3);
		for (uint i = 0; i < numV; ++i) {
			V(i, 0) = verts[i * 4];
			V(i, 1) = verts[i * 4+1];
			V(i, 2) = verts[i * 4+2];
		}
		for (int i = 0; i < numF; ++i) {
			F(i, 0) = i * 3;
			F(i, 1) = i * 3 + 1;
			F(i, 2) = i * 3 + 2;
		}
	}
};