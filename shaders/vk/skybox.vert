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

layout(set = 0, binding = 0) uniform UBO
{
	mat4 transform;
}
consts;

layout(location = 0) in vec3 inPosition;

layout(location = 0) out vec3 outPosition;

void
main()
{
	gl_Position = consts.transform * vec4(inPosition, 1.0);
	gl_Position.z = gl_Position.w;

	outPosition = inPosition;
}
