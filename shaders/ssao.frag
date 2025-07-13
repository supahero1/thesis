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

layout(constant_id = 0) const int SSAO_KERNEL_SIZE = 64;
layout(constant_id = 1) const float SSAO_RADIUS = 0.5;
layout(constant_id = 2) const float SSAO_BIAS = 0.025;
layout(constant_id = 3) const int SSAO_NOISE_SIZE = 8;

layout(set = 0, binding = 0) uniform sampler2D inNormalMap;
layout(set = 1, binding = 0) uniform sampler2D inDepthMap;
layout(set = 2, binding = 0) uniform sampler2D inNoiseMap;

layout(set = 3, binding = 0) uniform Constants
{
	mat4 projection;
	mat4 inverseProjection;
}
consts;

layout(set = 4, binding = 0) uniform Data
{
	vec4 samples[SSAO_KERNEL_SIZE];
}
data;

layout(location = 0) in vec2 inCoords;

layout(location = 0) out float outColor;

void
main()
{
	outColor = 0.3;
}
