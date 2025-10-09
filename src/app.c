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

#include <thesis/vk.h>
#include <thesis/xr.h>
#include <thesis/app.h>
#include <thesis/file.h>
#include <thesis/debug.h>
#include <thesis/options.h>
#include <thesis/alloc_ext.h>
#include <thesis/simulation.h>

#include <signal.h>
#include <stdlib.h>
#include <libgen.h>
#include <limits.h>
#include <unistd.h>


struct app
{
	simulation_t simulation;
	vk_t vk;
	xr_t xr;
	event_wait_state_t* stop_wait_state;
};


app_t
app_init(
	int argc,
	char** argv
	)
{
	app_t app = alloc_malloc(app, 1);
	assert_not_null(app);

	assert_ge(argc, 1);
	assert_not_null(argv);
	assert_not_null(argv[0]);

	char exe_path[PATH_MAX];
	realpath(argv[0], exe_path);
	char* dir = dirname(exe_path);

	int status = chdir(dir);
	hard_assert_eq(status, 0);

	hard_assert_true(dir_exists("assets"));
	hard_assert_true(dir_exists("shaders"));

	if(!dir_exists("cache"))
	{
		dir_create("cache");
	}

	if(!dir_exists("cache/vk"))
	{
		dir_create("cache/vk");
	}

	if(!dir_exists("cache/xr"))
	{
		dir_create("cache/xr");
	}

	sigset_t set;
	sigemptyset(&set);
	sigaddset(&set, SIGINT);
	pthread_sigmask(SIG_BLOCK, &set, NULL);

	global_options = options_init(argc, (void*) argv);

	app->simulation = simulation_init(
		(simulation_camera_t)
		{
			.pos = { -1000.0f, 200.0f, 0.0f },
			.angle = { 0.0f, 180.0f, 180.0f },
			.fov = 60.0f,
			.near = 10.0f
		},
		(simulation_light_t)
		{
			.pos = { 2500.0f, 500.0f, 5000.0f },
			.target = { 0.0f, 0.0f, 0.0f },
			.left = -5000.0f,
			.right = 1000.0f,
			.bottom = -1000.0f,
			.top = 300.0f,
			.near = 100.0f,
			.far = 20000.0f
		},
		// (simulation_light_t)
		// {
		// 	.pos = { 2500.0f, 500.0f, 5000.0f },
		// 	.target = { 0.0f, 0.0f, 0.0f },
		// 	.left = -5000.0f,
		// 	.right = 1000.0f,
		// 	.bottom = -800.0f,
		// 	.top = 500.0f,
		// 	.near = 100.0f,
		// 	.far = 20000.0f
		// },
		"assets/skybox-clouds-in-the-sky-spatial-io"
		);

	simulation_event_table_t* table = simulation_get_event_table(app->simulation);
	app->stop_wait_state = event_target_init_wait(&table->stop_target);

	simulation_add_entity(
		app->simulation,
		(simulation_entity_init_t)
		{
			.model_path = "assets/basketball_court_set/scene.gltf",
			// .model_path = "assets/basketball_court__low-poly/scene.gltf",
			.translation = { 0.0f, 0.0f, 0.0f },
			.rotation = { 0.0f, 0.0f, 0.0f },
			.scale = 1.0f,
			// .scale = 1000.0f,
			.dynamic = false
		}
		);

	for(uint32_t i = 0; i < 100; ++i)
	{
	simulation_add_entity(
		app->simulation,
		(simulation_entity_init_t)
		{
			.model_path = "assets/basketball_and1_xcelerate/scene.gltf",
			.translation = { -1000.0f, 200.0f + i * 50.0f, -100.0f },
			.rotation = { 0.0f, 0.0f, 0.0f },
			.scale = 130.0f,
			.dynamic = true
		}
		);
	}

	app->vk = vk_init(app->simulation);
	app->xr = xr_init(app->simulation);

	return app;
}


void
app_free(
	app_t app
	)
{
	assert_not_null(app);

	simulation_free(app->simulation);

	options_free(global_options);
	global_options = NULL;

	alloc_free(app, 1);
}


void
app_run(
	app_t app
	)
{
	assert_not_null(app);

	event_target_wait(app->stop_wait_state);
}
