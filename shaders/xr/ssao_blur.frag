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

layout(constant_id = 0) const float ssao_blur_radius = 5.0;
layout(constant_id = 1) const float ssao_blur_falloff = 1.9;
layout(constant_id = 2) const float ssao_blur_depth_tolerance = 256.0;

layout(set = 0, binding = 0) uniform sampler2D inViewPosition;
layout(set = 0, binding = 1) uniform sampler2D inViewNormal;
layout(set = 1, binding = 0) uniform sampler2D inSSAO;

layout(location = 0) in vec2 inCoords;

layout(location = 0) out float outOcclusion;

void
main()
{
	vec2 texSize = vec2(textureSize(inViewPosition, 0));
	vec2 texel = 1.0 / texSize;

	vec4 centerPos = texture(inViewPosition, inCoords);
	vec3 centerNormal = texture(inViewNormal, inCoords).xyz * 2.0 - 1.0;

	float sum = 0.0;
	float wsum = 0.0;

	for(float x = -ssao_blur_radius; x <= ssao_blur_radius; ++x)
	for(float y = -ssao_blur_radius; y <= ssao_blur_radius; ++y)
	{
		vec2 offset = vec2(x, y) * texel;
		vec2 sampleUV = inCoords + offset;

		vec4 samplePos = texture(inViewPosition, sampleUV);
		vec3 sampleNormal = texture(inViewNormal, sampleUV).xyz * 2.0 - 1.0;
		float sampleSSAO = texture(inSSAO, sampleUV).r;

		float gs = exp(-length(vec2(x, y)) / ssao_blur_falloff);
		float gd = exp(-abs(centerPos.w - samplePos.w) / ssao_blur_depth_tolerance);
		float gn = max(dot(centerNormal, sampleNormal), 0.0);

		float w = gs * gd * gn;
		sum += sampleSSAO * w;
		wsum += w;
	}

	outOcclusion = sum / max(wsum, 1e-6);
}
