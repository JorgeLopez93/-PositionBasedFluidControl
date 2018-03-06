#version 330

in vec3 fPosition;
in vec3 fTexCoord;


layout(location = 0 ) out vec4 fColor;

uniform sampler3D volume;
uniform float trfunc_delta = 0.01;
uniform vec3 camPos;
uniform float sample_step = 1.0 / 512.0;
uniform float val_threshold = 0.5;
uniform float isoValue = 4/255.0;

vec4 PhongLighting(vec3 L, vec3 N, vec3 V, float specPower, vec3 diffuseColor)
{
	float diffuse = max(dot(L,N),0.0);
	vec3 halfVec = normalize(L+V);
	float specular = pow(max(0.00001,dot(halfVec,N)),specPower) + 0.5;
	return vec4((diffuse*diffuseColor + specular),1.0);
}

vec3 GetGradient(vec3 uvw)
{
	vec3 s1, s2;

	s1.x = texture(volume, uvw-vec3(trfunc_delta,0.0,0.0)).x ;
	s2.x = texture(volume, uvw+vec3(trfunc_delta,0.0,0.0)).x ;

	s1.y = texture(volume, uvw-vec3(0.0,trfunc_delta,0.0)).x ;
	s2.y = texture(volume, uvw+vec3(0.0,trfunc_delta,0.0)).x ;

	s1.z = texture(volume, uvw-vec3(0.0,0.0,trfunc_delta)).x ;
	s2.z = texture(volume, uvw+vec3(0.0,0.0,trfunc_delta)).x ;

	return normalize((s1-s2)/2.0);
}

vec3 Bisection(vec3 left, vec3 right , float iso)
{
	for(int i=0;i<4;i++)
	{
		vec3 midpoint = (right + left) * 0.5;
		float cM = texture(volume, midpoint).x ;
		if(cM < iso)
			left = midpoint;
		else
			right = midpoint;
	}
	return vec3(right + left) * 0.5;
}

void main() {
  const float brightness = 50.0;

  vec3 geomDir = -normalize(camPos - fPosition);
  vec3 ray_dir = geomDir * sample_step;
  vec3 ray_pos = fTexCoord.xyz;
  vec3 pos111 = vec3(1.0, 1.0, 1.0);
  vec3 pos000 = vec3(0.0, 0.0, 0.0);

  float a = 0.0;
  vec4 color;
  bool stop = false, flag_Phong = true;
	float density = 0, density2 = 0;
  do
  {
      ray_pos += ray_dir ;

      stop = dot(sign(ray_pos-pos000),sign(pos111-ray_pos)) < 3.0;

  		if (stop)
  			break;

      density2 = texture(volume, ray_pos).r;

      a += density * sample_step * val_threshold * brightness;



			if( flag_Phong && (density - isoValue) < 0  && (density2 - isoValue) >= 0.0)  {

				vec3 xN = ray_pos;
				vec3 xF = ray_pos + ray_dir;
				vec3 tc = Bisection(xN, xF, isoValue);

				vec3 N = GetGradient(tc);
				vec3 V = geomDir;
				vec3 L =  -V;

				color =  PhongLighting(L,N,V,50, vec3(0.529, 0.807, 0.92));
				flag_Phong = false;
			}
			density = density2;
			if (a > 1.0){
				a = 1.0;
				break;
			}
  }
  while(true);

  if (a <= 0.01)
    discard;
  else{
    fColor = vec4(0.529, 0.807, 0.92, 1.0 )* (1.0 - a) + vec4(0.3, 0.3, 0.7, 1.0) * a;
    fColor *= color;

  }

}
