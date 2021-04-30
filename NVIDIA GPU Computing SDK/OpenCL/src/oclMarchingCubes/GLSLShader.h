#pragma once
#include <GL/glew.h>
#include <map>
#include <string>

using namespace std;

class GLSLShader
{
public:
	GLSLShader(void);
	~GLSLShader(void);	
	void LoadFromString(GLenum whichShader, const string& source);
	void LoadFromFile(GLenum whichShader, const string& filename);
	void CreateAndLinkProgram();
	void Use();
	void UnUse();
	void AddAttribute(const string& attribute);
	GLuint GLSLShader::getAttribute(const string& attribute);
	void AddUniform(const string& uniform);

	//An indexer that returns the location of the attribute/uniform
	GLuint operator[](const string& attribute);
	GLuint operator()(const string& uniform);
	void DeleteShaderProgram();


	// utility uniform functions
	// ------------------------------------------------------------------------
	void setBool(const std::string &name, bool value) const {
		glUniform1i(glGetUniformLocation(_program, name.c_str()), (int)value);
	}
	// ------------------------------------------------------------------------
	void setInt(const std::string &name, int value) const {
		glUniform1i(glGetUniformLocation(_program, name.c_str()), value);
	}
	void setUint(const std::string &name, unsigned int value) const {
		glUniform1ui(glGetUniformLocation(_program, name.c_str()), value);
	}
	// ------------------------------------------------------------------------
	void setFloat(const std::string &name, float value) const {
		glUniform1f(glGetUniformLocation(_program, name.c_str()), value);
	}
	// ------------------------------------------------------------------------
	void setVec2(const std::string &name, const float* vec2) const {
		glUniform2fv(glGetUniformLocation(_program, name.c_str()), 1, vec2);
	}
	void setVec2(const std::string &name, float x, float y) const {
		glUniform2f(glGetUniformLocation(_program, name.c_str()), x, y);
	}
	// ------------------------------------------------------------------------
	void setVec3(const std::string &name, const float* vec3) const {
		glUniform3fv(glGetUniformLocation(_program, name.c_str()), 1, vec3);
	}
	void setVec3(const std::string &name, float x, float y, float z) const {
		glUniform3f(glGetUniformLocation(_program, name.c_str()), x, y, z);
	}
	// ------------------------------------------------------------------------
	void setVec4(const std::string &name, const float* vec4) const {
		glUniform4fv(glGetUniformLocation(_program, name.c_str()), 1, vec4);
	}
	void setVec4(const std::string &name, float x, float y, float z, float w) {
		glUniform4f(glGetUniformLocation(_program, name.c_str()), x, y, z, w);
	}
	// ------------------------------------------------------------------------
	void setMat2(const std::string &name, const float* mat2) const {
		glUniformMatrix2fv(glGetUniformLocation(_program, name.c_str()), 1, GL_FALSE, mat2);
	}
	// ------------------------------------------------------------------------
	void setMat3(const std::string &name, const float* mat3) const {
		glUniformMatrix3fv(glGetUniformLocation(_program, name.c_str()), 1, GL_FALSE, mat3);
	}
	// ------------------------------------------------------------------------
	void setMat4(const std::string &name, const float* mat4) const {
		glUniformMatrix4fv(glGetUniformLocation(_program, name.c_str()), 1, GL_FALSE, mat4);
	}

private:
	enum ShaderType {VERTEX_SHADER, FRAGMENT_SHADER, GEOMETRY_SHADER};
	GLuint	_program;
	int _totalShaders;
	GLuint _shaders[3];//0->vertexshader, 1->fragmentshader, 2->geometryshader
	map<string,GLuint> _attributeList;
	map<string,GLuint> _uniformLocationList;
};	
