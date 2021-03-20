#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fNormal;
layout(location = 3) in vec3 FragPos;  

layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform sampler2D texSampler;

layout(binding = 2) uniform ShaderInfo
{
    vec3 lightColor;
    vec3 lightPos;
} shaderInfo;

layout(binding = 0) uniform UniformBufferObject 
{
    mat4 view;
    mat4 proj;
    vec3 viewPos;
} ubo;

void main() 
    {
    float ambientStrength = 0.05;
    float specularStrength = 0.5;

    vec3 ambient = ambientStrength * shaderInfo.lightColor;
    vec3 norm = normalize(fNormal);
    vec3 lightDir = normalize(shaderInfo.lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * shaderInfo.lightColor;

    vec3 viewDir = normalize(ubo.viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * shaderInfo.lightColor;

    outColor = vec4((ambient + diffuse + specular), 1.0) * texture(texSampler, fragTexCoord);
    }