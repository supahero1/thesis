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

layout(early_fragment_tests) in;

layout(push_constant) uniform Constants
{
	vec4 diffuse;
	vec4 ambient;
}
consts;

layout(set = 1, binding = 0) uniform sampler2D inTexture;
layout(set = 2, binding = 0) uniform sampler2DShadow inDepthMap;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inCoords;
layout(location = 3) in vec3 inView;
layout(location = 4) in vec3 inLight;
layout(location = 5) in vec4 inShadowCoords;

layout(location = 0) out vec4 outColor;

const vec2 poissonDisk[8] = vec2[](
	vec2(-0.326, -0.406),
	vec2(-0.840, -0.074),
	vec2(-0.696,  0.457),
	vec2(-0.203,  0.621),
	vec2( 0.962, -0.194),
	vec2( 0.473, -0.480),
	vec2( 0.519,  0.767),
	vec2( 0.185, -0.893)
	);

float
getShadow(
	vec4 shadowCoord
	)
{
	vec3 projCoord = shadowCoord.xyz / shadowCoord.w;
	if(projCoord.z > 1.0 || projCoord.z < 0.0)
	{
		return 1.0;
	}

	vec2 texelSize = 1.0 / textureSize(inDepthMap, 0);
	float shadow = 0.0;
	for(int i = 0; i < 8; ++i)
	{
		vec2 offset = poissonDisk[i] * texelSize;
		shadow += texture(inDepthMap, vec3(projCoord.xy + offset, projCoord.z - 0.001));
	}
	return max(shadow / 8.0, 0.5);
}

void
main()
{
	vec4 texel = texture(inTexture, inCoords);
	float shadow = getShadow(inShadowCoords);

	vec3 N = normalize(inNormal);
	vec3 L = normalize(inLight);
	float NL = dot(N, L);

	float wrap = 0.5;
	float diffuseFactor = clamp((NL + wrap) / (1.0 + wrap), 0.2, 1.0);
	vec3 diffuse = diffuseFactor * consts.diffuse.rgb;

	vec3 ambient = consts.ambient.rgb;

	vec3 lighting = (ambient + diffuse * shadow) * texel.rgb;
	outColor = vec4(lighting, texel.a);
}
