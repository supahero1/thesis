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

layout(constant_id = 0) const bool enable_ssao = true;

layout(set = 0, binding = 0) uniform sampler2DArray inScene;
layout(set = 1, binding = 0) uniform sampler2DArray inSSAO;

layout(location = 0) in vec2 inCoords;

layout(location = 0) out vec4 outColor;

void
main()
{
	vec4 color = texture(inScene, vec3(inCoords, gl_ViewIndex));
	vec4 ssao = texture(inSSAO, vec3(inCoords, gl_ViewIndex));

	if(enable_ssao)
	{
		color.rgb *= ssao.r;
	}

	outColor = color;
}
