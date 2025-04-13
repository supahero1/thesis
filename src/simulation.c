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

#include <thesis/debug.h>
#include <thesis/alloc_ext.h>
#include <thesis/simulation.h>

#include <string.h>
#include <stdatomic.h>


struct simulation
{
	vec3 camera;

	atomic_flag stopped;

	simulation_event_table_t event_table;
};


simulation_t
simulation_init(
	void
	)
{
	simulation_t simulation = alloc_malloc(sizeof(*simulation));
	assert_not_null(simulation);

	vec3 camera = { 0.0f, 0.0f, 0.0f };
	memcpy(simulation->camera, camera, sizeof(vec3));

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
