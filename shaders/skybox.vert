#version 450
#extension GL_ARB_separate_shader_objects : enable

//--------------------------------------------------------------------------------------------------
// Vertex shader to draw an evironment cube.
// - Z-Buffer should not be write into
// - Only the camera rotation is applied, not the translation
//

// Scene UBO
layout(set = 0, binding = 0) uniform UBOscene
{
  mat4  projection;
  mat4  modelView;
  vec4  camPos;
  vec4  lightDir;
  float lightIntensity;
  float exposure;
}
uboScene;


// Input
layout(location = 0) in vec3 inPos;
// Output
layout(location = 0) out vec3 outWorldPos;


out gl_PerVertex
{
  vec4 gl_Position;
};


void main()
{
  vec4 pos    = vec4(inPos.xyz, 1.0);
  gl_Position = uboScene.projection * pos;

  mat4 m      = inverse(uboScene.modelView);
  m[3][0]     = 0.0;
  m[3][1]     = 0.0;
  m[3][2]     = 0.0;
  outWorldPos = vec3(m * pos).xyz;
}
