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
	float far;
}
simulation_camera_t;

typedef struct simulation_entity_init
{
	const char* model_path;
	vec3 translation;
	vec3 rotation;
	bool dynamic;
}
simulation_entity_init_t;

typedef struct simulation_entity_data
{
	mat4 transform;
	uint32_t model_index;
}
simulation_entity_data_t;


extern simulation_t
simulation_init(
	simulation_camera_t camera
	);


extern void
simulation_free(
	simulation_t simulation
	);


extern simulation_event_table_t*
simulation_get_event_table(
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


extern model_t**
simulation_get_models(
	simulation_t simulation,
	uint32_t* model_count
	);


extern void
simulation_stop(
	simulation_t simulation
	);


extern void
simulation_update(
	simulation_t simulation,
	float delta
	);
