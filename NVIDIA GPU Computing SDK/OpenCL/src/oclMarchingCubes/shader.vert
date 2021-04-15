#version 330 core
layout(location = 0) in vec4 aPos;
layout(location = 1) in vec4 aNormal;

out vec3 FragPos;
out vec3 Normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
	mat4 VM = view * model;
	FragPos = vec3(VM * vec4(aPos.x, aPos.y, aPos.z, 1.0));
	Normal = mat3(transpose(inverse(VM))) * vec3(aNormal);
	gl_Position = projection * vec4(FragPos, 1.0);
}