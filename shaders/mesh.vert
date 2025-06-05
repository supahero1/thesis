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

layout(binding = 0) uniform UBO
{
	mat4 projection;
	mat4 view;
	mat4 light_transform;
	vec4 light_position;
}
consts;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inCoords;

layout(location = 3) in mat4 inTransform;

layout(location = 0) out vec3 outPosition;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec2 outCoords;
layout(location = 3) out vec3 outView;
layout(location = 4) out vec3 outLight;
layout(location = 5) out vec4 outShadowCoords;

const mat4 toClip = mat4(
	0.5, 0.0, 0.0, 0.0,
	0.0, 0.5, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	0.5, 0.5, 0.0, 1.0
	);

void
main()
{
	vec4 inPos = vec4(inPosition, 1.0);
	vec4 worldPos = inTransform * inPos;
	gl_Position = consts.projection * consts.view * worldPos;

	outPosition = worldPos.xyz;
	outNormal = mat3(inTransform) * inNormal;
	outCoords = inCoords;

	outLight = normalize(consts.light_position.xyz - worldPos.xyz);
	outView = -worldPos.xyz;
	outShadowCoords = (toClip * consts.light_transform) * worldPos;
}
