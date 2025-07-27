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
	float shininess;
	float shininess_strength;

	float near;
}
consts;

layout(set = 1, binding = 0) uniform sampler2D inTexture;

layout(location = 0) in vec2 inCoords;
layout(location = 1) in vec3 inPosition;
layout(location = 2) in vec3 inNormal;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec4 outAlbedo;
layout(location = 3) out vec3 outDiffuse;
layout(location = 4) out vec3 outAmbient;
layout(location = 5) out vec2 outShininess;

void
main()
{
	outPosition = vec4(inPosition, -consts.near / gl_FragCoord.z);
	outNormal = normalize(inNormal) * 0.5 + 0.5;
	outAlbedo = vec4(texture(inTexture, inCoords).rgb, 1.0);
	outDiffuse = consts.diffuse.rgb;
	outAmbient = consts.ambient.rgb;
	outShininess = vec2(consts.shininess, consts.shininess_strength);
}
