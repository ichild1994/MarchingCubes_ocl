#pragma once
//
//#include <glm/glm.hpp>
//#include <glm/gtc/matrix_transform.hpp>
//#include <glm/gtc/constants.hpp>

#include <math.h>
#include <vector>
#include "Matrices.h"

class camera {
private:
	Matrix4 m_viewMat;
	Matrix4 m_projMat;
	Vector3 m_camPos;
	float m_fov;
	float m_orthoScale;
	float m_winWidth;
	float m_winHeight;
	float m_camDistance;
	float m_znear;
	float m_zfar;
	
	enum {
		PERSPECTIVE = 0,
		ORTHO = 1,
	};
	int m_projMode;

	// euler Angles for rotations on Model!
	float rx; // X
	float ry; // Y 
	float rz; // Z
	float rotate_speed;
	float displace_speed;

public:
	camera() {
		m_projMode = PERSPECTIVE;
		reset();
	}
	
	void setWindowSize(float _w, float _h) {
		m_winWidth = _w;
		m_winHeight = _h;
		updateMatrix();
	}

	void setRotate(float x, float y, float z) {
		rx = x;
		ry = y;
		rz = z;
		updateMatrix();
	}

	void addRotateZ(float _z) {
		rz += rotate_speed*_z;
		if (rz > 360.0f) {
			rz -= 360.0f;
		}
		else if (rz < 0.0f) {
			rz += 360.0f;
		}
	}
	void addRotateY(float _y) {
		ry += rotate_speed*_y;
		if (ry > 360.0f) {
			ry -= 360.0f;
		}
		else if (ry < 0.0f) {
			ry += 360.0f;
		}
	}
	void addRotateX(float _x) {
		rx += rotate_speed*_x;
		if (rx > 360.0f) {
			rx -= 360.0f;
		}
		else if (rx < 0.0f) {
			rx += 360.0f;
		}
	}
	void addRotate(float _x, float _y, float _z) {
		//roll += rotate_speed*_roll;
		//pitch += rotate_speed*_pitch;
		//yaw += rotate_speed*_yaw;
		addRotateX(_x);
		addRotateY(_y);
		addRotateZ(_z);
		updateMatrix();
	}
	
	void setCamPos(float x, float y, float z) {
		m_camPos.x = x;
		m_camPos.y = y;
		m_camPos.z = z;
	}

	void addCamPos(float dx, float dy, float dz) {
		m_camPos += displace_speed*Vector3(dx, dy, dz);
		m_camPos.x = boundf(m_camPos.x, -2.0f, 2.0f);
		m_camPos.y = boundf(m_camPos.y, -2.0f, 2.0f);
		m_camPos.z = boundf(m_camPos.z, 0.0f, 20.0f);
		updateMatrix();
	}

	void setDistance(float dist) {
		m_camDistance = dist;
	}

	void setZplane(float _znear, float _zfar) {
		m_znear = _znear;
		m_zfar = _zfar;
	}

	void setOrthoScale(float s) {
		m_orthoScale = s;
	}

	void setProjMode(int flag) {
		if(flag == ORTHO || flag == PERSPECTIVE)
			m_projMode = flag;
	}

	inline float boundf(float x, float low, float high) {
		if (x < low) return low;
		if (x > high) return high;
		return x;
	}
	void addScale(float ds) {
		if (m_projMode == PERSPECTIVE) {
			m_camPos.z -= ds * 0.005f;
			m_camPos.z = boundf(m_camPos.z, 0.0f, 1.5f*m_camDistance);
		}
		else if (m_projMode == ORTHO) {
			m_orthoScale += ds * 0.01f;
			m_orthoScale = boundf(m_orthoScale, 0.1f, 8.0f);
		}
		updateMatrix();
	}

	void updateMatrix() {
		//viewMat = glm::mat4(1.0f);
		//viewMat = glm::translate(viewMat, glm::vec3(-camPos.x, -camPos.y, -camPos.z));
		//viewMat = glm::rotate(viewMat, glm::radians(roll), glm::vec3(1.0f, 0.0f, 0.0f));
		//viewMat = glm::rotate(viewMat, glm::radians(yaw), glm::vec3(0.0f, 0.0f, 1.0f));
		//viewMat = glm::rotate(viewMat, glm::radians(pitch), glm::vec3(0.0f, 1.0f, 0.0f));
		//viewMat = glm::rotate(viewMat, glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		m_viewMat.identity();
		m_viewMat.translate(-m_camPos.x, -m_camPos.y, -m_camPos.z);

		m_viewMat.rotateZ(rz);
		m_viewMat.rotateX(rx);
		m_viewMat.rotateY(ry);
		m_viewMat.rotateY(180.0f);
		if (m_projMode == ORTHO) {
			//viewMat = glm::scale(viewMat, glm::vec3(orthoScale));
			m_viewMat.scale(m_orthoScale);
		}

		if (m_projMode == ORTHO) {
			if (m_winWidth > m_winHeight)
				//projMat = glm::ortho(-2.0f*winWidth / winHeight, 2.0f*winWidth / winHeight, -2.0f, 2.0f, -(-camDistance + 1000.0f), -(-camDistance - 1000.0f));
				m_projMat.setOrtho(-2.0f*m_winWidth / m_winHeight, 2.0f*m_winWidth / m_winHeight, -2.0f, 2.0f, -(-m_camPos[2] + 1000.0f), -(-m_camPos[2] - 1000.0f));
				//m_projMat.setOrtho(-2.0f*m_winWidth / m_winHeight, 2.0f*m_winWidth / m_winHeight, -2.0f, 2.0f, m_znear, m_zfar);
			else
				//projMat = glm::ortho(-2.0f, 2.0f, -2.0f*winHeight / winWidth, 2.0f*winHeight / winWidth, -(-camDistance + 1000.0f), -(-camDistance - 1000.0f));
				m_projMat.setOrtho(-2.0f, 2.0f, -2.0f*m_winHeight / m_winWidth, 2.0f*m_winHeight / m_winWidth, -(-m_camPos[2] + 1000.0f), -(-m_camPos[2] - 1000.0f));
				//m_projMat.setOrtho(-2.0f*m_winWidth / m_winHeight, 2.0f*m_winWidth / m_winHeight, -2.0f, 2.0f, m_znear, m_zfar);
		}
		else if (m_projMode == PERSPECTIVE) {
			if (m_winWidth > m_winHeight)
				//projMat = glm::perspective(glm::radians(fov), winWidth / winHeight, -(-camDistance * 0.25f), -(-camDistance - camDistance));
				m_projMat.setPerspectiveY(m_fov, m_winWidth / m_winHeight, m_znear, m_zfar);
			else
				//projMat = glm::perspective(2.0f*atan(winHeight/winWidth*tan(glm::radians(0.5f*fov))), winWidth / winHeight, -(-camDistance * 0.25f), -(-camDistance - camDistance));
				m_projMat.setPerspectiveX(m_fov, m_winHeight / m_winWidth, m_znear, m_zfar);
		}
	}

	void reset() {
		m_fov = 60.0;
		m_winWidth = m_winHeight = 512.0;
		m_orthoScale = 1.0;
		rz = 0.0f;
		ry = -45.0;
		rx = 45.0f;
		rotate_speed = 1.0;
		displace_speed = 0.005;
		m_camDistance = 2.0;
		m_camPos = Vector3(0.0f, 0.0f, m_camDistance);
		updateMatrix();
	}

	void changeProjMode() {
		m_projMode = 1 - m_projMode;
		updateMatrix();
		//reset();
	}

	const float* getProjMatrixDataPtr() {
		//return glm::value_ptr(projMat);
		return m_projMat.get();
	}
	const float* getProjMatrixDataTransPosePtr() {
		//return glm::value_ptr(projMat);
		return m_projMat.getTranspose();
	}
	const float* getViewMatrixDataPtr() {
		//return glm::value_ptr(viewMat);
		return m_viewMat.get();
	}
	const float* getViewMatrixDataTransPosePtr() {
		//return glm::value_ptr(viewMat);
		return m_viewMat.getTranspose();
	}

	//glm::mat4 getProjMatrixData() {
	//	return projMat;
	//}
	//glm::mat4 getViewMatrixData() {
	//	return viewMat;
	//}
	Matrix4 getProjMat4() {
		return m_projMat;
	}
	Matrix4 getViewMat4() {
		return m_viewMat;
	}
	float getFov() {
		return m_fov;
	}
	void getWindowSize(float *width_, float *height_) {
		*width_ = m_winWidth;
		*height_ = m_winHeight;
	}
	float getZnear() {
		return m_znear;
	}
	Vector3 getPos() {
		return m_camPos;
	}
};