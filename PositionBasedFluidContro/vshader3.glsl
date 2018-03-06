
#version 130

in vec3 vPosition;
in vec3 vTexCoord;

varying out vec3 fTexCoord;
varying out vec3 fPosition;

//uniform mat4 mv_matrix;
//uniform mat4 proj_matrix;
uniform mat4 MVP;

//uniform vec3 light_pos = vec3(-10.0, -10.0, 100.0);

void main()
{
  vec4 vPosition4 = vec4(vPosition.x, vPosition.y, vPosition.z, 1.0);
  fPosition = vPosition;
  gl_Position = MVP * vPosition4;
  fTexCoord = (vPosition + vec3(0.5))/2;

}
