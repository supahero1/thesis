/*
 *   Copyright 2025 Franciszek Balcerak
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#version 450

layout(constant_id = 0) const bool enable_depth_shadows = true;
layout(constant_id = 1) const bool enable_backface_shadows = true;
layout(constant_id = 2) const bool enable_specular = true;
layout(constant_id = 3) const float shadow_value = 0.2;
layout(constant_id = 4) const float lambert_start_angle = 80.0;
layout(constant_id = 5) const bool enable_ssao = true;

layout(set = 0, binding = 0) uniform UBO
{
	mat4 projection;
	mat4 inverse_projection;
	mat4 view;
	mat4 inverse_view;
	mat4 light_transform;
	vec4 light_direction;
	vec4 camera_position;
}
consts;

layout(set = 1, binding = 0) uniform sampler2DShadow inShadow;
layout(set = 2, binding = 0) uniform sampler2D inNormal;
layout(set = 2, binding = 1) uniform sampler2D inAlbedo;
layout(set = 2, binding = 2) uniform sampler2D inLinearDepth;
layout(set = 2, binding = 3) uniform sampler2D inDiffuse;
layout(set = 2, binding = 4) uniform sampler2D inAmbient;
layout(set = 2, binding = 5) uniform sampler2D inShininess;
layout(set = 3, binding = 0) uniform sampler2D inSSAO;

layout(location = 0) in vec2 inCoords;

layout(location = 0) out vec4 outColor;

#define POISSON_DISK_SAMPLES 64
const vec2 poissonDisk[POISSON_DISK_SAMPLES] = vec2[](
	vec2(-0.898863, -0.279893), vec2(-0.730302,  0.592534),
	vec2(-0.505085, -0.835948), vec2(-0.252065,  0.033621),
	vec2(-0.169128,  0.787611), vec2(-0.015243, -0.428581),
	vec2( 0.052820,  0.941328), vec2( 0.176461, -0.065545),
	vec2( 0.239324, -0.840798), vec2( 0.316089,  0.640520),
	vec2( 0.402636, -0.493976), vec2( 0.508104,  0.222384),
	vec2( 0.589139, -0.734491), vec2( 0.672803,  0.865427),
	vec2( 0.742353, -0.219504), vec2( 0.810574,  0.472782),
	vec2( 0.871167, -0.608355), vec2( 0.916843,  0.076045),
	vec2( 0.940866,  0.710779), vec2( 0.963471, -0.166668),
	vec2( 0.985921,  0.395780), vec2( 0.999650, -0.449770),
	vec2(-0.994503,  0.082728), vec2(-0.957585, -0.370339),
	vec2(-0.845778,  0.424368), vec2(-0.704944, -0.638706),
	vec2(-0.552109,  0.133246), vec2(-0.435728, -0.871587),
	vec2(-0.301504,  0.686566), vec2(-0.117189, -0.329705),
	vec2(-0.061765,  0.932759), vec2( 0.053677, -0.093417),
	vec2( 0.125603, -0.730303), vec2( 0.198305,  0.424368),
	vec2( 0.282860, -0.490822), vec2( 0.380482,  0.817551),
	vec2( 0.443794,  0.187383), vec2( 0.499691, -0.999557),
	vec2( 0.584347, -0.224169), vec2( 0.655295,  0.640106),
	vec2( 0.725916, -0.582847), vec2( 0.771618,  0.038161),
	vec2( 0.819230,  0.370339), vec2( 0.869408, -0.347570),
	vec2( 0.903823,  0.170566), vec2( 0.947230, -0.751610),
	vec2( 0.970597,  0.485121), vec2( 0.996160, -0.081249),
	vec2(-0.999650, -0.081249), vec2(-0.970597,  0.485121),
	vec2(-0.947230, -0.751610), vec2(-0.903823,  0.170566),
	vec2(-0.869408, -0.347570), vec2(-0.819230,  0.370339),
	vec2(-0.771618,  0.038161), vec2(-0.725916, -0.582847),
	vec2(-0.655295,  0.640106), vec2(-0.584347, -0.224169),
	vec2(-0.499691, -0.999557), vec2(-0.443794,  0.187383),
	vec2(-0.380482,  0.817551), vec2(-0.282860, -0.490822),
	vec2(-0.198305,  0.424368), vec2(-0.125603, -0.730303)
	);

float
getShadow(
	vec4 shadowCoord
	)
{
	if(!enable_depth_shadows)
	{
		return 1.0;
	}

	vec3 projCoord = shadowCoord.xyz / shadowCoord.w;
	if(projCoord.z > 1.0 || projCoord.z < 0.0)
	{
		return 1.0;
	}

	vec2 texelSize = 1.0 / textureSize(inShadow, 0);
	float shadow = 0.0;
	for(int i = 0; i < POISSON_DISK_SAMPLES; ++i)
	{
		vec2 offset = poissonDisk[i] * texelSize;
		shadow += texture(inShadow, vec3(projCoord.xy + offset, projCoord.z));
	}
	return max(shadow / float(POISSON_DISK_SAMPLES), shadow_value);
}

vec3
reconstructPosition(
	float depth
	)
{
	vec2 ndc = inCoords * 2.0 - 1.0;
	vec4 clip = vec4(ndc, depth, 1.0);
	vec4 viewPos = consts.inverse_projection * clip;
	return viewPos.xyz / viewPos.w;
}

void
main()
{
	vec3 normal = texture(inNormal, inCoords).xyz * 2.0 - 1.0;
	vec3 albedo = texture(inAlbedo, inCoords).rgb;
	vec3 diffuse = texture(inDiffuse, inCoords).rgb;
	vec3 ambient = texture(inAmbient, inCoords).rgb;
	vec2 shininessData = texture(inShininess, inCoords).xy;
	float shininess = shininessData.x;
	float shininessStrength = shininessData.y;
	float ssao = texture(inSSAO, inCoords).r;

	float depth = texture(inLinearDepth, inCoords).r;
	vec3 fragPos = reconstructPosition(depth);

	vec3 N = normalize(normal);
	vec3 L = normalize((consts.view * consts.light_direction).xyz);
	vec3 V = normalize(-fragPos);
	vec3 R = reflect(-L, N);
	float NL = dot(N, L);

	vec4 worldPos = consts.inverse_view * vec4(fragPos, 1.0);
	vec4 shadowCoord = consts.light_transform * worldPos;
	float shadow = getShadow(shadowCoord);

	float angle = degrees(acos(NL));
	float lambertFactor = 1.0;
	if(NL <= 0.0)
	{
		lambertFactor = shadow_value;
	}
	else if(angle > lambert_start_angle)
	{
		lambertFactor = mix(1.0, shadow_value, (angle - lambert_start_angle) / (90.0 - lambert_start_angle));
	}

	float diffuseFactor = enable_backface_shadows ?
		min(NL <= 0.0 ? shadow_value : lambertFactor, shadow) : 1.0;
	float specularFactor = pow(max(dot(R, V), 0.0), shininess) * shininessStrength * 2.0;

	vec3 lighting = ambient + diffuse * diffuseFactor;
	if(enable_specular && diffuseFactor != shadow_value)
	{
		lighting += specularFactor * (diffuseFactor - shadow_value) / (1.0 - shadow_value);
	}

	if(enable_ssao)
	{
		lighting *= ssao;
	}

	outColor = vec4(lighting * albedo, 1.0);
}
