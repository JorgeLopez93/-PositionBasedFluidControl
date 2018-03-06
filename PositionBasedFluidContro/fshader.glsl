#version 120

in vec3 fNormal;
in vec3 fLight;
in vec3 fView;
in vec3 fTexCoord;


varying out vec4 fColor;

uniform samplerCube cubemap;
uniform int selector;

uniform vec3 light_color = vec3(0.75, 0.75, 0.75);
uniform vec3 ambient_color = vec3(0.3);

uniform float ka = 0.2;
uniform float kd = 1.0;
uniform float ks = 1.0;
uniform float n = 1;

void main() {
  if(selector == 1){
    vec3 nNormal = normalize(fNormal);
    vec3 nLight = normalize(fLight);
    vec3 nView = normalize(fView);

    vec3 R = reflect(-nLight, nNormal);

    vec3 ambient  = ka * ambient_color;
    vec3 diffuse  = kd * light_color * max(0.0, dot(nNormal, nLight));
    vec3 specular = ks * light_color * pow(max(0.0, dot(R, nView)), n);

    vec3 f_color = ambient + diffuse + specular;

    fColor = vec4(f_color, 1.0);
  }else{
    fColor = textureCube(cubemap, fTexCoord);
  }

}
