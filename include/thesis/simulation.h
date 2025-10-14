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

#include <thesis/event.h>
#include <thesis/model.h>
#include <thesis/stats.h>
#include <thesis/extent.h>


typedef struct simulation* simulation_t;

typedef struct simulation_stop_event_data
{
	simulation_t simulation;
}
simulation_stop_event_data_t;

typedef struct simulation_free_event_data
{
	simulation_t simulation;
}
simulation_free_event_data_t;

typedef struct simulation_event_table
{
	event_target_t stop_target;
	event_target_t free_target;
}
simulation_event_table_t;

typedef struct simulation_camera
{
	vec3 pos;
	vec3 angle;
	float fov;
	float near;
}
simulation_camera_t;

typedef struct simulation_light
{
	vec3 pos;
	vec3 target;
	float left;
	float right;
	float bottom;
	float top;
	float near;
	float far;
}
simulation_light_t;

typedef struct simulation_entity_init
{
	const char* model_path;
	vec3 translation;
	vec3 rotation;
	float scale;
	bool dynamic;
}
simulation_entity_init_t;

typedef struct simulation_entity_data
{
	mat4 transform;
	uint32_t model_idx;
}
simulation_entity_data_t;

typedef struct simulation_transform
{
	mat4 projection;
	mat4 inverse_projection;
	mat4 view;
	mat4 inverse_view;
	mat4 light_transform;
	vec4 light_direction;
}
simulation_transform_t;

typedef struct simulation_eye_pose
{
	vec4 rotation;
	vec3 position;
	vec4 fov;
}
simulation_eye_pose_t;

typedef struct simulation_vr_transform
{
	mat4 projection[2];
	mat4 inverse_projection[2];
	mat4 view[2];
	mat4 inverse_view[2];
	mat4 light_transform;
	vec4 light_direction;
}
simulation_vr_transform_t;

typedef struct simulation_texture
{
	void* data;
	uint32_t size;
	uint32_t width;
	uint32_t height;
	uint32_t layers;
	uint32_t levels;
}
simulation_texture_t;

typedef struct simulation_model_info
{
	model_t** models;
	uint32_t model_count;
	uint32_t material_count;
}
simulation_model_info_t;


extern simulation_t
simulation_init(
	simulation_camera_t camera,
	simulation_light_t light,
	const char* skybox_path
	);


extern void
simulation_free(
	simulation_t simulation
	);


extern simulation_event_table_t*
simulation_get_event_table(
	simulation_t simulation
	);


extern stats_t
simulation_get_stats(
	simulation_t simulation
	);


extern void
simulation_add_entity(
	simulation_t simulation,
	simulation_entity_init_t entity_init
	);


extern simulation_entity_data_t*
simulation_get_entity_data(
	simulation_t simulation,
	uint32_t* data_count
	);


extern void
simulation_free_entity_data(
	simulation_entity_data_t* data,
	uint32_t data_count
	);


extern simulation_camera_t
simulation_get_camera(
	simulation_t simulation
	);


extern simulation_transform_t
simulation_get_transform(
	simulation_t simulation,
	pair_t extent
	);


extern simulation_vr_transform_t
simulation_get_vr_transform(
	simulation_t simulation,
	pair_t extent,
	simulation_eye_pose_t left_eye,
	simulation_eye_pose_t right_eye
	);


extern simulation_model_info_t
simulation_get_model_info(
	simulation_t simulation
	);


extern simulation_texture_t*
simulation_get_texture(
	simulation_t simulation,
	str_t path,
	bool is_cube_map
	);


extern const str_t
simulation_get_skybox_path(
	simulation_t simulation
	);


extern float
simulation_get_scale(
	simulation_t simulation
	);


extern void
simulation_modify_velocity(
	simulation_t simulation,
	vec3 velocity
	);


extern void
simulation_modify_angle(
	simulation_t simulation,
	vec3 angle
	);


extern void
simulation_stop(
	simulation_t simulation
	);


extern void
simulation_update(
	simulation_t simulation
	);
