#version 330 core
out vec4 FragColor;	//fragment shader output
		
//input form the vertex shader
//interpolated colour to fragment shader
in vec3 Normal;
in vec3 FragPos;

struct Light {
	vec4 position;
	vec4 ambient;
	vec4 diffuse;
	vec4 specular;
};

struct Material {
	vec4 ambient;
	vec4 diffuse;
	vec4 specular;
	float shininess;
};

#define MAXLIGHTNUM 10
uniform int lightNum;
uniform Light lights[MAXLIGHTNUM];
uniform Material material;

uniform vec4 paintColor;

void main()
{
	//FragColor = vec4(objectColor, 1.0);
	//FragColor = material.diffuse;

	//FragColor = vec4(1.0,1.0,1.0,1.0);
	//// ambient
	//float ambientStrength = 0.6;
	//vec3 ambient = ambientStrength * lightColor;

	//// diffuse 
	//vec3 norm = normalize(Normal);
	////vec3 lightDir = normalize(lightPos - FragPos);
	//vec3 lightDir = vec3(0.0, 0.0, -1.0);
	//float diff = max(dot(norm, lightDir), 0.0);
	//vec3 diffuse = diff * vec3(lights[0].diffuse);
	//FragColor = vec4(diffuse, 1.0f) + vec4(0.2,0.2,0.2,0.0);
	//// specular
	//float specularStrength = 0.5;
	//vec3 viewPos(0.0, 0.0, 0.0);
	//vec3 viewDir = normalize(viewPos - FragPos);
	//vec3 reflectDir = reflect(-lightDir, norm);
	//float spec = pow(max(dot(viewDir, reflectDir), 0.0f), 2);
	//vec3 specular = specularStrength * spec * lightColor;

	//vec3 result = (ambient + 0.5*diffuse + specular) * objectColor;
	//FragColor = vec4(result, 1.0);

	vec4 resColor = vec4(0.0, 0.0, 0.0, 0.0);
	vec3 viewPos = vec3(0.0, 0.0, 0.0);
	vec3 viewDir = normalize(viewPos - FragPos);

	vec3 norm = normalize(Normal);
	for (int i = 0; i < lightNum; ++i) 
	{
		vec4 lightpos = lights[i].position;
		// parallel light
		if (abs(lightpos.w) < 1e-6) {
			vec3 lightDir = normalize(vec3(lightpos));
			vec3 reflectDir = reflect(lightDir, norm);
			// ambient
			resColor += material.ambient * lights[i].ambient;
			// diffuse
			float kd = max(dot(norm, lightDir), 0.0);
			resColor += material.diffuse * lights[i].diffuse * kd;
			// specular
			float ks = max(dot(viewDir, reflectDir), 0.0);
			if (ks > 0) ks = pow(ks, exp2(material.shininess));
			resColor += material.specular * lights[i].specular * ks;
		}
		// point light
		else {
			vec3 lightDir = normalize(vec3(lightpos)-FragPos);
			vec3 reflectDir = reflect(lightDir, norm);
			// ambient
			resColor += material.ambient * lights[i].ambient;
			// diffuse
			float kd = max(dot(norm, lightDir), 0.0);
			resColor += material.diffuse * lights[i].diffuse * kd;
			// specular
			float ks = max(dot(viewDir, reflectDir), 0.0);
			if (ks > 0) ks = pow(ks, exp2(material.shininess));
			resColor += material.specular * lights[i].specular * ks;
		}
	}
	resColor.w = 1.0;
	FragColor = resColor;
	FragColor *= paintColor;
}
