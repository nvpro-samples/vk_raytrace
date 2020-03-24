#version 450
layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outFragColor;

layout(set = 0, binding = 0) uniform sampler2D inTxt;

void main()
{
  outFragColor = texture(inTxt, inUV).rgba;
}
