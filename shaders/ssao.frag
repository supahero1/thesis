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

layout(constant_id = 0) const int ssao_kernel_size = 32;
layout(constant_id = 1) const int ssao_noise_size = 4;
layout(constant_id = 2) const float ssao_radius = 32.0;
layout(constant_id = 3) const float ssao_bias = 0.05;
layout(constant_id = 4) const float ssao_power = 2.0;
layout(constant_id = 5) const float ssao_depth_k = 0.0025;
layout(constant_id = 6) const float ssao_depth_gamma = 1.5;
layout(constant_id = 7) const bool ssao_debug = false;

layout(set = 0, binding = 0) uniform sampler2D inViewPosition;
layout(set = 0, binding = 1) uniform sampler2D inViewNormal;
layout(set = 1, binding = 0) uniform sampler2D inNoise;

layout(set = 2, binding = 0) uniform UBO
{
	mat4 projection;
}
consts;

layout(set = 3, binding = 0) uniform KernelUBO
{
	vec4 samples[ssao_kernel_size];
}
data;

layout(location = 0) in vec2 inCoords;

layout(location = 0) out float outOcclusion;

void
main()
{
	vec4 fragPos = texture(inViewPosition, inCoords);
	vec3 normal = texture(inViewNormal, inCoords).xyz * 2.0 - 1.0;

	ivec2 texelCoords = ivec2(gl_FragCoord.xy);
	ivec2 noiseDim = ivec2(ssao_noise_size);
	ivec2 noiseTexelCoords = texelCoords % noiseDim;
	vec2 noiseCoords = (vec2(noiseTexelCoords) + 0.5) / vec2(noiseDim);
	vec3 randomVec = texture(inNoise, noiseCoords).xyz * 2.0 - 1.0;

	vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
	vec3 bitangent = cross(normal, tangent);
	mat3 tbn = mat3(tangent, bitangent, normal);

	float d = abs(fragPos.z);
	float depthFactor = 1.0 - exp(-ssao_depth_k * d);
	depthFactor = pow(depthFactor, ssao_depth_gamma);
	int current_kernel_size = max(4, int(float(ssao_kernel_size) * depthFactor));

	float occlusion = 0.0;
	for(int i = 0; i < current_kernel_size; ++i)
	{
		vec3 samplePos = tbn * data.samples[i].xyz;
		samplePos = fragPos.xyz + samplePos * ssao_radius;

		vec4 offset = consts.projection * vec4(samplePos, 1.0);
		offset /= offset.w;
		vec2 offsetNDC = offset.xy * 0.5 + 0.5;

		float sampleDepth = texture(inViewPosition, offsetNDC).w;
		float rangeCheck = smoothstep(0.0, 1.0, ssao_radius / abs(fragPos.z - sampleDepth));
		occlusion += (sampleDepth < samplePos.z - ssao_bias ? 1.0 : 0.0) * rangeCheck;
	}

	occlusion = 1.0 - occlusion / float(current_kernel_size);
	outOcclusion = pow(occlusion, ssao_power);
	if(fragPos.z == 0.0)
	{
		outOcclusion = 0.0;
	}

	if(ssao_debug)
	{
		outOcclusion = float(current_kernel_size) / float(ssao_kernel_size);
	}
}
