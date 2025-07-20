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

layout(constant_id = 0) const int ssao_kernel_size = 64;
layout(constant_id = 1) const int ssao_noise_size = 8;
layout(constant_id = 2) const float ssao_radius = 0.5;
layout(constant_id = 3) const float ssao_bias = 0.025;
layout(constant_id = 4) const float ssao_power = 2.2;

layout(set = 0, binding = 0) uniform sampler2D inNormal;
layout(set = 0, binding = 2) uniform sampler2D inLinearDepth;
layout(set = 1, binding = 0) uniform sampler2D inNoise;

layout(set = 2, binding = 0) uniform UBO
{
	mat4 projection;
	mat4 inverse_projection;
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
	float fragLinearDepth = texture(inLinearDepth, inCoords).r;
	vec3 normal = normalize(texture(inNormal, inCoords).rgb);

	vec2 ndc = inCoords * 2.0 - 1.0;
	vec4 clip = vec4(ndc, fragLinearDepth, 1.0);
	vec4 viewPos = consts.inverse_projection * clip;
	vec3 fragPos = viewPos.xyz / viewPos.w;

	ivec2 screenSize = textureSize(inLinearDepth, 0);
	vec2 noiseCoords = vec2(screenSize) / vec2(ssao_noise_size) * inCoords;
	vec3 randomVec = normalize(texture(inNoise, noiseCoords).rgb);

	vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
	vec3 bitangent = cross(normal, tangent);
	mat3 tbn = mat3(tangent, bitangent, normal);

	float occlusion = 0.0;
	for(int i = 0; i < ssao_kernel_size; ++i)
	{
		vec3 samplePos = tbn * data.samples[i].xyz;
		samplePos = fragPos + samplePos * ssao_radius;

		vec4 offsetClip = consts.projection * vec4(samplePos, 1.0);
		offsetClip /= offsetClip.w;
		vec2 offsetNDC = offsetClip.xy * 0.5 + 0.5;

		float sampleSceneDepth = texture(inLinearDepth, offsetNDC).r;
		float samplePointLinearDepth = -samplePos.z;
		float rangeCheck = smoothstep(0.0, 1.0, ssao_bias / abs(fragLinearDepth - sampleSceneDepth));

		occlusion += (sampleSceneDepth < samplePointLinearDepth - ssao_bias ? 1.0 : 0.0) * rangeCheck;
	}

	occlusion = 1.0 - occlusion / float(ssao_kernel_size);
	outOcclusion = pow(occlusion, ssao_power);
}
