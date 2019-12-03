#version 450
layout(location = 0) in vec2 outUV;
layout(location = 0) out vec4 fragColor;

layout(set = 0, binding = 0) uniform sampler2D noisyTxt;

layout(push_constant) uniform shaderInformation
{
  float aspectRatio;
}
pushc;

void main()
{
  vec2 uv   = outUV;
  fragColor = texture(noisyTxt, uv).rgba;
}
