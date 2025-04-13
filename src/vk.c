/*
 *   Copyright 2025 Franciszek Balcerak
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by vklicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <thesis/vk.h>
#include <thesis/debug.h>
#include <thesis/window.h>
#include <thesis/threads.h>
#include <thesis/alloc_ext.h>

#include <signal.h>


struct vk
{
	event_listener_t* window_close_once_listener;
	event_listener_t* window_key_down_listener;

	window_manager_t window_manager;
	window_t window;
	thread_t window_thread;

	simulation_t simulation;
};


private void
vk_window_close_once_fn(
	vk_t vk,
	window_close_event_data_t* event_data
	)
{
	vk->window_close_once_listener = NULL;
	window_close(vk->window);
}


private void
vk_window_free_once_fn(
	vk_t vk,
	window_free_event_data_t* event_data
	)
{
	window_event_table_t* table = window_get_event_table(vk->window);

	event_target_del(&table->key_down_target, vk->window_key_down_listener);

	if(vk->window_close_once_listener)
	{
		event_target_del_once(&table->close_target, vk->window_close_once_listener);
	}

	simulation_stop(vk->simulation);
}


private void
vk_window_init_once_fn(
	vk_t vk,
	window_init_event_data_t* event_data
	)
{
	// window_toggle_fullscreen(vk->window);
	window_show(vk->window);
}


private void
vk_window_key_down_fn(
	vk_t vk,
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
		window_close(vk->window);
	}
}


private void
vk_window_thread_fn(
	vk_t vk
	)
{
	assert_not_null(vk);

	sigset_t set;
	sigemptyset(&set);
	sigaddset(&set, SIGINT);
	pthread_sigmask(SIG_UNBLOCK, &set, NULL);

	window_manager_run(vk->window_manager);
}


private void
vk_init_window(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk->window_manager = window_manager_init();
	vk->window = window_init();
	window_manager_add(vk->window_manager, vk->window, "Thesis", NULL);

	window_event_table_t* table = window_get_event_table(vk->window);

	event_listener_data_t close_once_data =
	{
		.fn = (void*) vk_window_close_once_fn,
		.data = vk
	};
	vk->window_close_once_listener = event_target_once(&table->close_target, close_once_data);

	event_listener_data_t free_once_data =
	{
		.fn = (void*) vk_window_free_once_fn,
		.data = vk
	};
	event_target_once(&table->free_target, free_once_data);

	event_listener_data_t init_once_data =
	{
		.fn = (void*) vk_window_init_once_fn,
		.data = vk
	};
	event_target_once(&table->init_target, init_once_data);

	event_listener_data_t key_down_data =
	{
		.fn = (void*) vk_window_key_down_fn,
		.data = vk
	};
	vk->window_key_down_listener = event_target_add(&table->key_down_target, key_down_data);


	thread_data_t thread_data =
	{
		.fn = (void*) vk_window_thread_fn,
		.data = vk
	};
	thread_init(&vk->window_thread, thread_data);
}


private void
vk_free_window(
	vk_t vk
	)
{
	assert_not_null(vk);

	window_manager_free(vk->window_manager);
}


private void
vk_init_internal(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_init_window(vk);
}


private void
vk_free_internal(
	vk_t vk
	)
{
	assert_not_null(vk);

	vk_free_window(vk);
}


private void
vk_free(
	vk_t vk,
	simulation_free_event_data_t* event_data
	)
{
	assert_not_null(vk);

	window_manager_stop_running(vk->window_manager);
	thread_join(vk->window_thread);

	vk_free_internal(vk);

	alloc_free(vk, sizeof(*vk));
}


vk_t
vk_init(
	simulation_t simulation
	)
{
	assert_not_null(simulation);

	vk_t vk = alloc_malloc(sizeof(*vk));
	assert_not_null(vk);

	vk->simulation = simulation;

	simulation_event_table_t* table = simulation_get_event_table(vk->simulation);

	event_listener_data_t free_data =
	{
		.fn = (void*) vk_free,
		.data = vk
	};
	event_target_once(&table->free_target, free_data);

	vk_init_internal(vk);

	return vk;
}
