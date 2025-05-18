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
#include <thesis/app.h>
#include <thesis/file.h>
#include <thesis/debug.h>
#include <thesis/options.h>
#include <thesis/alloc_ext.h>
#include <thesis/simulation.h>

#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <libgen.h>
#include <limits.h>
#include <unistd.h>


struct app
{
	simulation_t simulation;
	vk_t vk;
};


app_t
app_init(
	int argc,
	char** argv
	)
{
	app_t app = alloc_malloc(sizeof(*app));
	assert_ptr(app, sizeof(*app));

	assert_ge(argc, 1);
	assert_not_null(argv);
	assert_not_null(argv[0]);

	char exe_path[PATH_MAX];
	realpath(argv[0], exe_path);
	char* dir = dirname(exe_path);

	int status = chdir(dir);
	hard_assert_eq(status, 0);

	sigset_t set;
	sigemptyset(&set);
	sigaddset(&set, SIGINT);
	pthread_sigmask(SIG_BLOCK, &set, NULL);

	global_options = options_init(argc, (void*) argv);

	app->simulation = simulation_init(
		(simulation_camera_t)
		{
			.pos = { 0.0f, 0.0f, 0.0f },
			.angle = { 0.0f, 0.0f, 0.0f },
			.fov = 90.0f,
			.near = 1.0f,
			.far = 10000.0f
		}
		);

	simulation_add_entity(
		app->simulation,
		(simulation_entity_init_t)
		{
			.model_path = "assets/ccity-building-set-1/maya2sketchfab.fbx",
			.translation = { 0.0f, 0.0f, 0.0f },
			.rotation = { 0.0f, 0.0f, 0.0f },
			.dynamic = false
		}
		);

	app->vk = vk_init(app->simulation);

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

	alloc_free(app, sizeof(*app));
}


void
app_run(
	app_t app
	)
{
	assert_not_null(app);

	simulation_event_table_t* table = simulation_get_event_table(app->simulation);
	event_target_wait(&table->stop_target);
}
