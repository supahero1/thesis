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

#extension GL_EXT_multiview : require

layout(set = 0, binding = 0) uniform UBO
{
	mat4 projection[2];
	mat4 view[2];
	mat4 light_transform;
	vec4 light_direction;
	vec4 camera_position[2];
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
layout(location = 6) out vec3 outViewPosition;
layout(location = 7) out vec3 outViewNormal;

const mat4 toClip = mat4(
	0.5, 0.0, 0.0, 0.0,
	0.0, 0.5, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	0.5, 0.5, 0.0, 1.0
	);

void
main()
{
	vec4 worldPos = inTransform * vec4(inPosition, 1.0);
	gl_Position = consts.projection[gl_ViewIndex] * consts.view[gl_ViewIndex] * worldPos;

	outPosition = worldPos.xyz;
	outNormal = mat3(inTransform) * inNormal;
	outCoords = inCoords;

	outLight = normalize(consts.light_direction.xyz);
	outView = normalize(consts.camera_position[gl_ViewIndex].xyz - worldPos.xyz);
	outShadowCoords = (toClip * consts.light_transform) * worldPos;

	outViewPosition = vec3(consts.view[gl_ViewIndex] * worldPos);
	outViewNormal = mat3(consts.view[gl_ViewIndex]) * outNormal;
}
