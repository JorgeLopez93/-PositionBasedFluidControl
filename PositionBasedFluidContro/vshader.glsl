
#version 120

in vec3 vPosition;

varying out vec3 fTexCoord;

uniform mat4 mv_matrix;
uniform mat4 proj_matrix;

uniform vec3 light_pos = vec3(-10.0, -10.0, 100.0);

void main()
{
  vec4 vPosition4 = vec4(vPosition.x, vPosition.y, vPosition.z, 1.0);

  gl_Position = proj_matrix * mv_matrix * vPosition4;
  fTexCoord = vPosition;

}
