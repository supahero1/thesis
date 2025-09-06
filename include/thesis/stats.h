/*
 *   Copyright 2024-2025 Franciszek Balcerak
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

#include <stdint.h>


typedef struct stats* stats_t;


extern stats_t
stats_init(
	void
	);


extern void
stats_free(
	stats_t stats
	);


extern void
stats_add(
	stats_t stats,
	const char* name,
	uint32_t max_count
	);


extern void
stats_del(
	stats_t stats,
	const char* name
	);


extern void
stats_log(
	stats_t stats,
	const char* name,
	uint64_t time
	);
