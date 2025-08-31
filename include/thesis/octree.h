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

#include <thesis/extent_3d.h>

/* Above or equal, if can, 1 node splits into 4 */
#define OCTREE_SPLIT_THRESHOLD 15

/* Below or equal, if can, 4 nodes merge into 1 */
#define OCTREE_MERGE_THRESHOLD 13

#define OCTREE_DEDUPE_COLLISIONS 1

/* Do not modify unless you know what you are doing. Use octree.min_size. */
#define OCTREE_MAX_DEPTH 20

/* Do not modify */
#define OCTREE_DFS_LENGTH (OCTREE_MAX_DEPTH * 7 + 1)


typedef struct octree_node
{
	int32_t count;

	union
	{
		uint32_t next;

		struct
		{
			uint32_t head;
			uint32_t position_flags;
		};

		uint32_t heads[8];
	};
}
octree_node_t;


typedef struct octree_node_entity
{
	uint32_t next;
	uint32_t entity;
}
octree_node_entity_t;


#ifndef octree_entity_data


	typedef struct octree_entity_data
	{
		rect_extent_3d_t rect_extent;
	}
	octree_entity_data_t;


	#define octree_entity_data octree_entity_data_t
	#define octree_get_entity_data_rect_extent(entity) (entity).rect_extent
#endif


typedef struct octree_entity
{
	union
	{
		octree_entity_data data;
		uint32_t next;
	};

	uint32_t query_tick;
	uint8_t update_tick;
	bool fully_in_node;
}
octree_entity_t;


#define octree_get_entity_rect_extent(entity)	\
octree_get_entity_data_rect_extent((entity)->data)


typedef struct octree_node_info
{
	uint32_t node_idx;
	half_extent_3d_t extent;
}
octree_node_info_t;


typedef struct octree_ht_entry
{
	uint32_t next;
	uint32_t idx[2];
}
octree_ht_entry_t;


typedef struct octree_removal
{
	uint32_t entity_idx;
}
octree_removal_t;


typedef struct octree_node_removal
{
	uint32_t node_idx;
	uint32_t node_entity_idx;
	uint32_t prev_node_entity_idx;
}
octree_node_removal_t;


typedef struct octree_insertion
{
	octree_entity_data data;
}
octree_insertion_t;


typedef struct octree_reinsertion
{
	uint32_t entity_idx;
}
octree_reinsertion_t;


typedef enum octree_status
{
	OCTREE_STATUS_CHANGED,
	OCTREE_STATUS_NOT_CHANGED,
	kOCTREE_STATUS
}
octree_status_t;


typedef struct octree octree_t;


typedef void
(*octree_query_fn_t)(
	octree_t* ot,
	uint32_t entity_idx,
	octree_entity_data* entity
	);


typedef void
(*octree_node_query_fn_t)(
	octree_t* ot,
	const octree_node_info_t* info
	);


typedef void
(*octree_collide_fn_t)(
	const octree_t* ot,
	octree_entity_data* entity_a,
	octree_entity_data* entity_b
	);


typedef octree_status_t
(*octree_update_fn_t)(
	octree_t* ot,
	uint32_t entity_idx,
	octree_entity_data* entity
	);


struct octree
{
	octree_node_t* nodes;
	octree_node_entity_t* node_entities;
	octree_entity_t* entities;
#if OCTREE_DEDUPE_COLLISIONS == 1
	octree_ht_entry_t* ht_entries;
#endif
	octree_removal_t* removals;
	octree_node_removal_t* node_removals;
	octree_insertion_t* insertions;
	octree_reinsertion_t* reinsertions;

	uint32_t nodes_used;
	uint32_t nodes_size;

	uint32_t node_entities_used;
	uint32_t node_entities_size;

	uint32_t entities_used;
	uint32_t entities_size;

	uint32_t ht_entries_used;
	uint32_t ht_entries_size;

	uint32_t removals_used;
	uint32_t removals_size;

	uint32_t node_removals_used;
	uint32_t node_removals_size;

	uint32_t insertions_used;
	uint32_t insertions_size;

	uint32_t reinsertions_used;
	uint32_t reinsertions_size;

	uint32_t query_tick;
	uint8_t update_tick;

	rect_extent_3d_t rect_extent;
	half_extent_3d_t half_extent;

	float min_size;
};


extern void
octree_init(
	octree_t* ot
	);


extern void
octree_free(
	octree_t* ot
	);


extern void
octree_insert(
	octree_t* ot,
	const octree_entity_data* data
	);


extern void
octree_remove(
	octree_t* ot,
	uint32_t entity_idx
	);


extern void
octree_normalize(
	octree_t* ot
	);


extern void
octree_update(
	octree_t* ot,
	octree_update_fn_t update_fn
	);


extern void
octree_query(
	octree_t* ot,
	rect_extent_3d_t extent,
	octree_query_fn_t query_fn
	);


extern void
octree_query_nodes(
	octree_t* ot,
	rect_extent_3d_t extent,
	octree_node_query_fn_t node_query_fn
	);


extern void
octree_collide(
	octree_t* ot,
	octree_collide_fn_t collide_fn
	);
