#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 normals;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out vec3 fNormal;
layout(location = 3) out vec3 FragPos;  

layout(binding = 0) uniform UniformBufferObject 
{
    mat4 view;
    mat4 proj;
    vec3 viewPos;
} ubo;

layout( push_constant ) uniform CameraMatrix {
    mat4 model;
} PushConstant;

void main() {
    gl_Position = ubo.proj * ubo.view * PushConstant.model * vec4(inPosition, 1.0);
    fragColor = inColor;
    fragTexCoord = inTexCoord;
    fNormal = mat3(transpose(inverse(PushConstant.model))) * normals;
    FragPos = vec3(PushConstant.model * vec4(inPosition, 1.0));
}