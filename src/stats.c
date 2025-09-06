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
#include <thesis/time.h>
#include <thesis/debug.h>
#include <thesis/stats.h>
#include <thesis/alloc_ext.h>

#include <stdio.h>


typedef struct stats_entry
{
	uint64_t* times;
	uint32_t idx;
	uint32_t count;
	uint32_t added_count;
	uint32_t max_count;
}
stats_entry_t;

struct stats
{
	sync_mtx_t mutex;
	hash_table_t table;

	char* buffer;
	uint32_t buffer_smaller_idx;
	uint32_t buffer_used;
	uint32_t buffer_size;

	time_timers_t timers;
	time_timer_t log_timer;
};


private void
stats_hash_table_value_free_fn(
	stats_entry_t* entry
	)
{
	assert_not_null(entry);

	alloc_free(entry->times, sizeof(*entry->times) * entry->max_count);
	alloc_free(entry, sizeof(*entry));
}


private void
stats_hash_table_for_each_fn(
	str_t key,
	stats_entry_t* entry,
	stats_t stats
	)
{
	assert_not_null(key);
	assert_not_null(entry);
	assert_not_null(stats);

	if(!entry->added_count)
	{
		return;
	}
	entry->added_count = 0;

	uint64_t total = 0;
	for(uint32_t i = 0; i < entry->count; ++i)
	{
		total += entry->times[i];
	}

	if(stats->buffer_used + 128 > stats->buffer_size)
	{
		uint32_t new_size = MACRO_NEXT_OR_EQUAL_POWER_OF_2(stats->buffer_used + 128);
		stats->buffer = alloc_remalloc(stats->buffer, stats->buffer_size, new_size);
		assert_not_null(stats->buffer);

		stats->buffer_size = new_size;
		stats->buffer_smaller_idx = 0;
	}

	int written = snprintf(
		stats->buffer + stats->buffer_used,
		stats->buffer_size - stats->buffer_used,
		"%s: %.2fms\n", (char*) key->str, (double) total / entry->count / time_ms_to_ns(1)
		);
	assert_gt(written, 0);

	stats->buffer_used += written;
}


private void
stats_log_timer_fn(
	stats_t stats
	)
{
	assert_not_null(stats);

	stats->buffer_used = 0;

	sync_mtx_lock(&stats->mutex);
	hash_table_for_each(stats->table, (void*) stats_hash_table_for_each_fn, stats);
	sync_mtx_unlock(&stats->mutex);

	if(stats->buffer_used > 0)
	{
		assert_lt(stats->buffer_used, stats->buffer_size);
		stats->buffer[stats->buffer_used++] = '\0';
		printf("\n---- STATS ----\n%s\n", stats->buffer);
	}

	if(stats->buffer_used < stats->buffer_size / 4)
	{
		++stats->buffer_smaller_idx;
	}

	if(stats->buffer_smaller_idx > 3)
	{
		uint32_t new_size = stats->buffer_size >> 1;
		stats->buffer = alloc_remalloc(stats->buffer, stats->buffer_size, new_size);
		assert_not_null(stats->buffer);

		stats->buffer_size = new_size;
		stats->buffer_smaller_idx = 0;
	}
}


stats_t
stats_init(
	void
	)
{
	stats_t stats = alloc_calloc(sizeof(*stats));
	assert_not_null(stats);

	stats->table = hash_table_init(64, NULL, (void*) stats_hash_table_value_free_fn);

	stats->timers = time_timers_init();
	time_timer_init(&stats->log_timer);
	time_timers_add_interval(stats->timers,
		(time_interval_t)
		{
			.timer = &stats->log_timer,
			.data =
			{
				.fn = (void*) stats_log_timer_fn,
				.data = stats
			},
			.base_time = time_get_with_sec(-2),
			.interval = time_sec_to_ns(3),
			.count = 1
		}
		);

	return stats;
}


void
stats_free(
	stats_t stats
	)
{
	assert_not_null(stats);

	time_timers_free(stats->timers);

	alloc_free(stats->buffer, stats->buffer_size);

	hash_table_free(stats->table);

	alloc_free(stats->buffer, stats->buffer_size);

	alloc_free(stats, sizeof(*stats));
}


void
stats_add(
	stats_t stats,
	const char* name,
	uint32_t max_count
	)
{
	assert_not_null(stats);
	assert_not_null(name);
	assert_gt(max_count, 0);

	stats_entry_t* entry = alloc_malloc(sizeof(*entry));
	assert_not_null(entry);

	uint64_t* times = alloc_malloc(sizeof(*entry->times) * max_count);
	assert_not_null(times);

	*entry =
	(stats_entry_t)
	{
		.times = times,
		.idx = 0,
		.count = 0,
		.added_count = 0,
		.max_count = max_count
	};

	sync_mtx_lock(&stats->mutex);

	bool added = hash_table_add(stats->table, name, entry);
	assert_true(added);

	sync_mtx_unlock(&stats->mutex);
}


void
stats_del(
	stats_t stats,
	const char* name
	)
{
	assert_not_null(stats);
	assert_not_null(name);

	sync_mtx_lock(&stats->mutex);

	bool removed = hash_table_del(stats->table, name);
	assert_true(removed);

	sync_mtx_unlock(&stats->mutex);
}


void
stats_log(
	stats_t stats,
	const char* name,
	uint64_t time
	)
{
	assert_not_null(stats);
	assert_not_null(name);
	assert_ge(time, 0);

	if(!time)
	{
		return;
	}

	sync_mtx_lock(&stats->mutex);

	stats_entry_t* entry = hash_table_get(stats->table, name);
	assert_not_null(entry);

	entry->times[entry->idx] = time;
	entry->idx = (entry->idx + 1) % entry->max_count;
	++entry->added_count;
	entry->count = MACRO_MIN(entry->count + 1, entry->max_count);

	sync_mtx_unlock(&stats->mutex);
}
