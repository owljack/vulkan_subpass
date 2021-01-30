#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 1) uniform sampler2D texSampler;
layout (input_attachment_index = 0, set = 0, binding = 2) uniform subpassInput inputColor;
layout(location = 0) out vec4 outColor;

void main() 
    {
    outColor = vec4(1.0,0.0,0.5,1.0) * subpassLoad(inputColor);
    }