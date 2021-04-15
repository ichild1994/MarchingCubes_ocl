#version 330 core
out vec4 FragColor;	//fragment shader output
		
//input form the vertex shader
//interpolated colour to fragment shader
in vec3 Normal;
in vec3 FragPos;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;
uniform vec3 objectColor;


void main()
{
	//FragColor = vec4(1.0, 0.0, 0.0, 1.0);

	// ambient
	float ambientStrength = 0.6;
	vec3 ambient = ambientStrength * lightColor;

	// diffuse 
	vec3 norm = normalize(Normal);
	vec3 lightDir = normalize(lightPos - FragPos);
	//vec3 lightDir = vec3(0.0, 0.0, -1.0);
	float diff = max(dot(norm, lightDir), 0.0);
	vec3 diffuse = diff * lightColor;

	// specular
	float specularStrength = 0.5;
	vec3 viewDir = normalize(viewPos - FragPos);
	vec3 reflectDir = reflect(-lightDir, norm);
	float spec = pow(max(dot(viewDir, reflectDir), 0.0f), 2);
	vec3 specular = specularStrength * spec * lightColor;

	vec3 result = (ambient + 0.5*diffuse + specular) * objectColor;
	FragColor = vec4(result, 1.0);
}