

#version 120

in vec3 vPosition;
in vec3 vNormal;
in vec3 vColor;

varying out vec3 color;
varying out vec3 fPosition;
varying out vec3 fNormal;
varying out vec3 fLight;
varying out vec3 fView;

uniform mat4 mv_matrix;
uniform mat4 view_matrix;
uniform mat4 proj_matrix;
uniform vec4 eye;

uniform vec3 light_pos = vec3( 3.0, 3.0, -3.0);

void main()
{
  vec4 vPosition4 = vec4(vPosition.x, vPosition.y, vPosition.z, 1.0);
  vec4 P = mv_matrix*vPosition4;
  fNormal = mat3(mv_matrix) * vNormal;
  fLight = light_pos - P.xyz;
  fView = eye.xyz  - P.xyz;
  color = vColor;
  gl_Position = proj_matrix * mv_matrix * vPosition4;
  fPosition = vPosition;
}
