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

#include <thesis/macro.h>
#include <thesis/stats.h>
#include <thesis/extent_3d.h>


typedef struct collider* collider_t;

typedef enum collider_entity_type
{
	COLLIDER_ENTITY_TYPE_TRIANGLE,
	COLLIDER_ENTITY_TYPE_SPHERE,
	MACRO_ENUM_BITS(COLLIDER_ENTITY_TYPE)
}
collider_entity_type_t;

typedef struct collider_entity
{
	rect_extent_3d_t rect_extent;
	collider_entity_type_t type;
	rect_extent_3d_t* external;
}
collider_entity_t;


extern collider_t
collider_init(
	stats_t stats
	);


extern void
collider_free(
	collider_t collider
	);


extern void
collider_add(
	collider_t collider,
	const collider_entity_t* entity
	);


extern void
collider_update(
	collider_t collider,
	float delta
	);
