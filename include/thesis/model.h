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

#pragma once

#include <thesis/str.h>
#include <thesis/macro.h>

#define CGLM_FORCE_RADIANS
#define CGLM_FORCE_LEFT_HANDED
#define CGLM_FORCE_DEPTH_ZERO_TO_ONE
#define CGLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <cglm/cglm.h>


typedef struct material
{
	str_t texture;
	vec4 diffuse;
	vec4 ambient;
}
material_t;


typedef struct mesh
{
	uint32_t material_idx;
	uint32_t vertex_count;
	vec3* vertices;
	vec3* normals;
	vec2* coords;

	uint32_t* indexes;
	uint32_t index_count;
}
mesh_t;


typedef struct model
{
	material_t* materials;
	mesh_t* meshes;

	uint32_t material_count;
	uint32_t mesh_count;
}
model_t;


extern model_t*
model_init(
	const char* path
	);


extern void
model_free(
	model_t* model
	);
