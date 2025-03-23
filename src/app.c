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

#include <thesis/app.h>
#include <thesis/file.h>
#include <thesis/debug.h>
#include <thesis/graphics.h>
#include <thesis/alloc_ext.h>

#include <stdio.h>
#include <stdlib.h>
#include <libgen.h>
#include <limits.h>
#include <unistd.h>


struct app
{
	window_event_table_t* window_event_table;
	event_listener_t* window_close_once_listener;
	event_listener_t* window_key_down_listener;

	window_manager_t manager;
	window_t window;
	graphics_t graphics;
};


private void
app_window_close_once_fn(
	app_t app,
	window_close_event_data_t* event_data
	)
{
	app->window_close_once_listener = NULL;
	window_close(event_data->window);
}


private void
app_window_free_once_fn(
	app_t app,
	window_free_event_data_t* event_data
	)
{
	window_event_table_t* table = app->window_event_table;

	event_target_del(&table->key_down_target, app->window_key_down_listener);

	if(app->window_close_once_listener)
	{
		event_target_del_once(&table->close_target, app->window_close_once_listener);
	}
}


private void
app_window_init_once_fn(
	app_t app,
	window_init_event_data_t* event_data
	)
{
	window_toggle_fullscreen(app->window);
}


private void
app_window_key_down_fn(
	app_t app,
	window_key_down_event_data_t* event_data
	)
{
	if(
		(
			(event_data->mods & WINDOW_MOD_CTRL_BIT) &&
			(
				event_data->key == WINDOW_KEY_W ||
				event_data->key == WINDOW_KEY_R
				)
			) ||
		event_data->key == WINDOW_KEY_ESCAPE ||
		event_data->key == WINDOW_KEY_F11
		)
	{
		window_close(app->window);
	}
}


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

	app->manager = window_manager_init();
	app->window = window_init();
	window_manager_add(app->manager, app->window);
	window_show(app->window);

	app->window_event_table = window_get_event_table(app->window);
	window_event_table_t* table = app->window_event_table;

	event_listener_data_t close_once_data =
	{
		.fn = (event_fn_t) app_window_close_once_fn,
		.data = app
	};
	app->window_close_once_listener = event_target_once(&table->close_target, close_once_data);

	event_listener_data_t free_once_data =
	{
		.fn = (event_fn_t) app_window_free_once_fn,
		.data = app
	};
	event_target_once(&table->free_target, free_once_data);

	event_listener_data_t init_once_data =
	{
		.fn = (event_fn_t) app_window_init_once_fn,
		.data = app
	};
	event_target_once(&table->init_target, init_once_data);

	event_listener_data_t key_down_data =
	{
		.fn = (event_fn_t) app_window_key_down_fn,
		.data = app
	};
	app->window_key_down_listener = event_target_add(&table->key_down_target, key_down_data);

	app->graphics = graphics_init(app->window);

	return app;
}


void
app_free(
	app_t app
	)
{
	assert_not_null(app);

	window_manager_free(app->manager);

	alloc_free(app, sizeof(*app));
}


void
app_run(
	app_t app
	)
{
	assert_not_null(app);

	window_manager_run(app->manager);
}
