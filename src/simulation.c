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

#include <thesis/hash.h>
#include <thesis/debug.h>
#include <thesis/alloc_ext.h>
#include <thesis/simulation.h>

#include <stdatomic.h>


typedef struct simulation_entity
{
	uint32_t model_index;
	vec3 translation;
	vec3 rotation;
	bool dynamic;
}
simulation_entity_t;

struct simulation
{
	simulation_camera_t camera;

	model_t** models;
	uint32_t model_count;

	hash_table_t model_table;

	simulation_entity_t* entities;
	uint32_t entity_count;

	atomic_flag stopped;

	simulation_event_table_t event_table;
};


simulation_t
simulation_init(
	simulation_camera_t camera
	)
{
	simulation_t simulation = alloc_malloc(sizeof(*simulation));
	assert_not_null(simulation);

	simulation->camera = camera;

	simulation->models = NULL;
	simulation->model_count = 0;

	simulation->model_table = hash_table_init(8, NULL, NULL);

	simulation->entities = NULL;
	simulation->entity_count = 0;

	atomic_flag_clear(&simulation->stopped);

	event_target_init(&simulation->event_table.free_target);

	return simulation;
}


void
simulation_free(
	simulation_t simulation
	)
{
	assert_not_null(simulation);

	simulation_free_event_data_t event_data =
	{
		.simulation = simulation
	};
	event_target_fire(&simulation->event_table.free_target, &event_data);

	event_target_free(&simulation->event_table.free_target);

	alloc_free(simulation->entities, sizeof(*simulation->entities) * simulation->entity_count);

	hash_table_free(simulation->model_table);

	for(uint32_t i = 0; i < simulation->model_count; ++i)
	{
		model_free(simulation->models[i]);
	}

	alloc_free(simulation->models, sizeof(*simulation->models) * simulation->model_count);

	alloc_free(simulation, sizeof(*simulation));
}


simulation_event_table_t*
simulation_get_event_table(
	simulation_t simulation
	)
{
	assert_not_null(simulation);

	return &simulation->event_table;
}


void
simulation_add_entity(
	simulation_t simulation,
	simulation_entity_init_t entity_init
	)
{
	assert_not_null(simulation);

	simulation->entities = alloc_remalloc(
		simulation->entities,
		sizeof(*simulation->entities) * simulation->entity_count,
		sizeof(*simulation->entities) * (simulation->entity_count + 1)
		);
	assert_not_null(simulation->entities);

	simulation_entity_t* entity = &simulation->entities[simulation->entity_count++];

	uintptr_t model = (uintptr_t) hash_table_get(
		simulation->model_table,
		entity_init.model_path
		);

	if(!model)
	{
		simulation->models = alloc_remalloc(
			simulation->models,
			sizeof(*simulation->models) * simulation->model_count,
			sizeof(*simulation->models) * (simulation->model_count + 1)
			);
		assert_not_null(simulation->models);

		simulation->models[simulation->model_count++] =
			model_init(entity_init.model_path);

		hash_table_set(
			simulation->model_table,
			entity_init.model_path,
			(void*) (uintptr_t) simulation->model_count
			);
		model = simulation->model_count;
	}

	entity->model_index = --model;

	glm_vec3_copy(entity_init.translation, entity->translation);
	glm_vec3_copy(entity_init.rotation, entity->rotation);
	entity->dynamic = entity_init.dynamic;
}


simulation_entity_data_t*
simulation_get_entity_data(
	simulation_t simulation,
	uint32_t* data_count
	)
{
	assert_not_null(simulation);

	if(data_count)
	{
		*data_count = simulation->entity_count;
	}

	simulation_entity_data_t* data = alloc_malloc(
		sizeof(*data) * simulation->entity_count
		);
	assert_not_null(data);

	for(uint32_t i = 0; i < simulation->entity_count; ++i)
	{
		simulation_entity_data_t* cur_data = &data[i];
		simulation_entity_t* entity = &simulation->entities[i];

		cur_data->model_index = entity->model_index;

		glm_mat4_identity(cur_data->transform);
		glm_translate(cur_data->transform, entity->translation);
		glm_rotate_x(cur_data->transform, entity->rotation[0], cur_data->transform);
		glm_rotate_y(cur_data->transform, entity->rotation[1], cur_data->transform);
		glm_rotate_z(cur_data->transform, entity->rotation[2], cur_data->transform);
	}

	return data;
}


model_t**
simulation_get_models(
	simulation_t simulation,
	uint32_t* model_count
	)
{
	assert_not_null(simulation);

	if(model_count)
	{
		*model_count = simulation->model_count;
	}

	return simulation->models;
}


void
simulation_stop(
	simulation_t simulation
	)
{
	assert_not_null(simulation);

	bool stopped = atomic_flag_test_and_set(&simulation->stopped);
	if(stopped)
	{
		return;
	}

	simulation_stop_event_data_t event_data =
	{
		.simulation = simulation
	};
	event_target_fire(&simulation->event_table.stop_target, &event_data);
}


void
simulation_update(
	simulation_t simulation,
	float delta
	)
{
	assert_not_null(simulation);

	(void) delta;
}
