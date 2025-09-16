/* skip */
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
#include <thesis/octree.h>
#include <thesis/alloc_ext.h>

#include <string.h>


void
octree_init(
	octree_t* ot
	)
{
	ot->nodes = alloc_malloc(sizeof(*ot->nodes));
	ot->node_entities = NULL;
	ot->entities = NULL;
#if OCTREE_DEDUPE_COLLISIONS == 1
	ot->ht_entries = NULL;
#endif
	ot->removals = NULL;
	ot->node_removals = NULL;
	ot->insertions = NULL;
	ot->reinsertions = NULL;

	ot->nodes_used = 1;
	ot->nodes_size = 1;

	ot->node_entities_used = 1;
	ot->node_entities_size = 0;

	ot->entities_used = 1;
	ot->entities_size = 0;

#if OCTREE_DEDUPE_COLLISIONS == 1
	ot->ht_entries_used = 1;
	ot->ht_entries_size = 0;
#endif

	ot->removals_used = 0;
	ot->removals_size = 0;

	ot->node_removals_used = 0;
	ot->node_removals_size = 0;

	ot->insertions_used = 0;
	ot->insertions_size = 0;

	ot->reinsertions_used = 0;
	ot->reinsertions_size = 0;

	assert_not_null(ot->nodes);

	ot->nodes[0].head = 0;
	ot->nodes[0].count = 0;
	ot->nodes[0].position_flags = 0b111111; /* TLRBFB */
}


void
octree_free(
	octree_t* ot
	)
{
	alloc_free(ot->nodes, sizeof(*ot->nodes) * ot->nodes_size);
	alloc_free(ot->node_entities, sizeof(*ot->node_entities) * ot->node_entities_size);
	alloc_free(ot->entities, sizeof(*ot->entities) * ot->entities_size);
#if OCTREE_DEDUPE_COLLISIONS == 1
	alloc_free(ot->ht_entries, sizeof(*ot->ht_entries) * ot->ht_entries_size);
#endif
	alloc_free(ot->removals, sizeof(*ot->removals) * ot->removals_size);
	alloc_free(ot->node_removals, sizeof(*ot->node_removals) * ot->node_removals_size);
	alloc_free(ot->insertions, sizeof(*ot->insertions) * ot->insertions_size);
	alloc_free(ot->reinsertions, sizeof(*ot->reinsertions) * ot->reinsertions_size);
}


#define octree_descend(_extent)							\
do														\
{														\
	float half_w = info.extent.w * 0.5f;				\
	float half_h = info.extent.h * 0.5f;				\
	float half_d = info.extent.d * 0.5f;				\
														\
	if(_extent.min_x <= info.extent.x)					\
	{													\
		if(_extent.min_y <= info.extent.y)				\
		{												\
			if(_extent.min_z <= info.extent.z)			\
			{											\
				*(node_info++) =						\
				(octree_node_info_t)					\
				{										\
					.node_idx = node->heads[0],			\
					.extent =							\
					(half_extent_3d_t)					\
					{									\
						.x = info.extent.x - half_w,	\
						.y = info.extent.y - half_h,	\
						.z = info.extent.z - half_d,	\
						.w = half_w,					\
						.h = half_h,					\
						.d = half_d						\
					}									\
				};										\
			}											\
			if(_extent.max_z >= info.extent.z)			\
			{											\
				*(node_info++) =						\
				(octree_node_info_t)					\
				{										\
					.node_idx = node->heads[1],			\
					.extent =							\
					(half_extent_3d_t)					\
					{									\
						.x = info.extent.x - half_w,	\
						.y = info.extent.y - half_h,	\
						.z = info.extent.z + half_d,	\
						.w = half_w,					\
						.h = half_h,					\
						.d = half_d						\
					}									\
				};										\
			}											\
		}												\
		if(_extent.max_y >= info.extent.y)				\
		{												\
			if(_extent.min_z <= info.extent.z)			\
			{											\
				*(node_info++) =						\
				(octree_node_info_t)					\
				{										\
					.node_idx = node->heads[2],			\
					.extent =							\
					(half_extent_3d_t)					\
					{									\
						.x = info.extent.x - half_w,	\
						.y = info.extent.y + half_h,	\
						.z = info.extent.z - half_d,	\
						.w = half_w,					\
						.h = half_h,					\
						.d = half_d						\
					}									\
				};										\
			}											\
			if(_extent.max_z >= info.extent.z)			\
			{											\
				*(node_info++) =						\
				(octree_node_info_t)					\
				{										\
					.node_idx = node->heads[3],			\
					.extent =							\
					(half_extent_3d_t)					\
					{									\
						.x = info.extent.x - half_w,	\
						.y = info.extent.y + half_h,	\
						.z = info.extent.z + half_d,	\
						.w = half_w,					\
						.h = half_h,					\
						.d = half_d						\
					}									\
				};										\
			}											\
		}												\
	}													\
	if(_extent.max_x >= info.extent.x)					\
	{													\
		if(_extent.min_y <= info.extent.y)				\
		{												\
			if(_extent.min_z <= info.extent.z)			\
			{											\
				*(node_info++) =						\
				(octree_node_info_t)					\
				{										\
					.node_idx = node->heads[4],			\
					.extent =							\
					(half_extent_3d_t)					\
					{									\
						.x = info.extent.x + half_w,	\
						.y = info.extent.y - half_h,	\
						.z = info.extent.z - half_d,	\
						.w = half_w,					\
						.h = half_h,					\
						.d = half_d						\
					}									\
				};										\
			}											\
			if(_extent.max_z >= info.extent.z)			\
			{											\
				*(node_info++) =						\
				(octree_node_info_t)					\
				{										\
					.node_idx = node->heads[5],			\
					.extent =							\
					(half_extent_3d_t)					\
					{									\
						.x = info.extent.x + half_w,	\
						.y = info.extent.y - half_h,	\
						.z = info.extent.z + half_d,	\
						.w = half_w,					\
						.h = half_h,					\
						.d = half_d						\
					}									\
				};										\
			}											\
		}												\
		if(_extent.max_y >= info.extent.y)				\
		{												\
			if(_extent.min_z <= info.extent.z)			\
			{											\
				*(node_info++) =						\
				(octree_node_info_t)					\
				{										\
					.node_idx = node->heads[6],			\
					.extent =							\
					(half_extent_3d_t)					\
					{									\
						.x = info.extent.x + half_w,	\
						.y = info.extent.y + half_h,	\
						.z = info.extent.z - half_d,	\
						.w = half_w,					\
						.h = half_h,					\
						.d = half_d						\
					}									\
				};										\
			}											\
			if(_extent.max_z >= info.extent.z)			\
			{											\
				*(node_info++) =						\
				(octree_node_info_t)					\
				{										\
					.node_idx = node->heads[7],			\
					.extent =							\
					(half_extent_3d_t)					\
					{									\
						.x = info.extent.x + half_w,	\
						.y = info.extent.y + half_h,	\
						.z = info.extent.z + half_d,	\
						.w = half_w,					\
						.h = half_h,					\
						.d = half_d						\
					}									\
				};										\
			}											\
		}												\
	}													\
}														\
while(0)


void
octree_insert(
	octree_t* ot,
	const octree_entity_data* data
	)
{
	if(ot->insertions_used >= ot->insertions_size)
	{
		uint32_t new_size = (ot->insertions_used << 1) | 1;

		ot->insertions = alloc_remalloc(ot->insertions,
			sizeof(*ot->insertions) * ot->insertions_size,
			sizeof(*ot->insertions) * new_size);
		assert_not_null(ot->insertions);

		ot->insertions_size = new_size;
	}

	uint32_t insertion_idx = ot->insertions_used++;
	octree_insertion_t* insertion = ot->insertions + insertion_idx;

	insertion->data = *data;
}


void
octree_remove(
	octree_t* ot,
	uint32_t entity_idx
	)
{
	if(ot->removals_used >= ot->removals_size)
	{
		uint32_t new_size = (ot->removals_used << 1) | 1;

		ot->removals = alloc_remalloc(ot->removals,
			sizeof(*ot->removals) * ot->removals_size,
			sizeof(*ot->removals) * new_size);
		assert_not_null(ot->removals);

		ot->removals_size = new_size;
	}

	uint32_t removal_idx = ot->removals_used++;
	octree_removal_t* removal = ot->removals + removal_idx;

	removal->entity_idx = entity_idx;
}


void
octree_normalize(
	octree_t* ot
	)
{
	if(!ot->insertions_used && !ot->reinsertions_used && !ot->removals_used && !ot->node_removals_used)
	{
		return;
	}

	octree_node_t* nodes = ot->nodes;
	octree_node_entity_t* node_entities = ot->node_entities;
	octree_entity_t* entities = ot->entities;

	uint32_t free_node_entity = 0; /* Not reflected in ot->free_node_entity */
	uint32_t node_entities_used = ot->node_entities_used;
	uint32_t node_entities_size = ot->node_entities_size;

	uint32_t free_entity = 0; /* Not reflected in ot->free_entity */
	uint32_t entities_used = ot->entities_used;
	uint32_t entities_size = ot->entities_size;

	octree_node_info_t node_infos[OCTREE_DFS_LENGTH];
	octree_node_info_t* node_info;


	if(ot->node_removals_used)
	{
		octree_node_removal_t* node_removals = ot->node_removals;
		octree_node_removal_t* node_removal = node_removals + ot->node_removals_used - 1;

		while(node_removal >= node_removals)
		{
			uint32_t node_idx = node_removal->node_idx;
			octree_node_t* node = nodes + node_idx;

			uint32_t node_entity_idx = node_removal->node_entity_idx;
			octree_node_entity_t* node_entity = node_entities + node_entity_idx;

			uint32_t prev_node_entity_idx = node_removal->prev_node_entity_idx;
			octree_node_entity_t* prev_node_entity = node_entities + prev_node_entity_idx;

			if(prev_node_entity_idx)
			{
				prev_node_entity->next = node_entity->next;
			}
			else
			{
				node->head = node_entity->next;
			}

			--node->count;

			node_entity->next = free_node_entity;
			free_node_entity = node_entity_idx;

			--node_removal;
		}

		alloc_free(node_removals, sizeof(*node_removals) * ot->node_removals_size);
		ot->node_removals = NULL;
		ot->node_removals_used = 0;
		ot->node_removals_size = 0;
	}


	{
		octree_reinsertion_t* reinsertions = ot->reinsertions;
		octree_reinsertion_t* reinsertion = reinsertions;
		octree_reinsertion_t* reinsertion_end = reinsertion + ot->reinsertions_used;

		while(reinsertion != reinsertion_end)
		{
			uint32_t entity_idx = reinsertion->entity_idx;
			octree_entity_t* entity = entities + entity_idx;

			rect_extent_3d_t entity_extent = octree_get_entity_rect_extent(entity);

			node_info = node_infos;

			*(node_info++) =
			(octree_node_info_t)
			{
				.node_idx = 0,
				.extent = ot->half_extent
			};

			do
			{
				octree_node_info_t info = *(--node_info);
				octree_node_t* node = nodes + info.node_idx;

				if(node->count == -1)
				{
					octree_descend(entity_extent);
					continue;
				}

				uint32_t node_entity_idx = node->head;
				octree_node_entity_t* node_entity;

				while(node_entity_idx)
				{
					node_entity = node_entities + node_entity_idx;

					if(node_entity->entity == entity_idx)
					{
						goto goto_skip;
					}

					node_entity_idx = node_entity->next;
				}

				if(free_node_entity)
				{
					node_entity_idx = free_node_entity;
					node_entity = node_entities + node_entity_idx;
					free_node_entity = node_entity->next;
				}
				else
				{
					if(node_entities_used >= node_entities_size)
					{
						uint32_t new_size = (node_entities_used << 1) | 1;

						node_entities = alloc_remalloc(node_entities,
							sizeof(*node_entities) * node_entities_size,
							sizeof(*node_entities) * new_size);
						assert_not_null(node_entities);

						node_entities_size = new_size;
					}

					node_entity_idx = node_entities_used++;
					node_entity = node_entities + node_entity_idx;
				}

				node_entity->next = node->head;
				node_entity->entity = entity_idx;
				node->head = node_entity_idx;

				++node->count;

				goto_skip:;
			}
			while(node_info != node_infos);

			++reinsertion;
		}

		alloc_free(reinsertions, sizeof(*reinsertions) * ot->reinsertions_size);
		ot->reinsertions = NULL;
		ot->reinsertions_used = 0;
		ot->reinsertions_size = 0;
	}


	{
		octree_removal_t* removals = ot->removals;
		octree_removal_t* removal = removals;
		octree_removal_t* removal_end = removal + ot->removals_used;

		while(removal != removal_end)
		{
			node_info = node_infos;

			*(node_info++) =
			(octree_node_info_t)
			{
				.node_idx = 0,
				.extent = ot->half_extent
			};

			uint32_t entity_idx = removal->entity_idx;
			octree_entity_t* entity = entities + entity_idx;
			rect_extent_3d_t entity_extent = octree_get_entity_rect_extent(entity);

			do
			{
				octree_node_info_t info = *(--node_info);
				octree_node_t* node = nodes + info.node_idx;

				if(node->count == -1)
				{
					octree_descend(entity_extent);
					continue;
				}

				octree_node_entity_t* prev_node_entity = NULL;
				uint32_t node_entity_idx = node->head;

				while(node_entity_idx)
				{
					octree_node_entity_t* node_entity = node_entities + node_entity_idx;

					if(node_entity->entity == entity_idx)
					{
						if(prev_node_entity)
						{
							prev_node_entity->next = node_entity->next;
						}
						else
						{
							node->head = node_entity->next;
						}

						--node->count;

						node_entity->next = free_node_entity;
						free_node_entity = node_entity_idx;

						break;
					}

					prev_node_entity = node_entity;
					node_entity_idx = node_entity->next;
				}
			}
			while(node_info != node_infos);

			entity->next = free_entity;
			free_entity = entity_idx;

			++removal;
		}

		alloc_free(removals, sizeof(*removals) * ot->removals_size);
		ot->removals = NULL;
		ot->removals_used = 0;
		ot->removals_size = 0;
	}


	{
		octree_insertion_t* insertions = ot->insertions;
		octree_insertion_t* insertion = insertions;
		octree_insertion_t* insertion_end = insertion + ot->insertions_used;

		while(insertion != insertion_end)
		{
			octree_entity_data* data = &insertion->data;

			uint32_t entity_idx;
			octree_entity_t* entity;

			if(free_entity)
			{
				entity_idx = free_entity;
				entity = entities + entity_idx;
				free_entity = entity->next;
			}
			else
			{
				if(entities_used >= entities_size)
				{
					uint32_t new_size = (entities_used << 1) | 1;

					entities = alloc_remalloc(entities,
						sizeof(*entities) * entities_size,
						sizeof(*entities) * new_size);
					assert_not_null(entities);

					entities_size = new_size;
				}

				entity_idx = entities_used++;
				entity = entities + entity_idx;
			}

			entity->data = *data;
			entity->query_tick = ot->query_tick;
			entity->update_tick = ot->update_tick;

			rect_extent_3d_t entity_extent = octree_get_entity_rect_extent(entity);

			node_info = node_infos;

			*(node_info++) =
			(octree_node_info_t)
			{
				.node_idx = 0,
				.extent = ot->half_extent
			};

			do
			{
				octree_node_info_t info = *(--node_info);
				octree_node_t* node = nodes + info.node_idx;

				if(node->count == -1)
				{
					octree_descend(entity_extent);
					continue;
				}

				uint32_t node_entity_idx;
				octree_node_entity_t* node_entity;

				if(free_node_entity)
				{
					node_entity_idx = free_node_entity;
					node_entity = node_entities + node_entity_idx;
					free_node_entity = node_entity->next;
				}
				else
				{
					if(node_entities_used >= node_entities_size)
					{
						uint32_t new_size = (node_entities_used << 1) | 1;

						node_entities = alloc_remalloc(node_entities,
							sizeof(*node_entities) * node_entities_size,
							sizeof(*node_entities) * new_size);
						assert_not_null(node_entities);

						node_entities_size = new_size;
					}

					node_entity_idx = node_entities_used++;
					node_entity = node_entities + node_entity_idx;
				}

				node_entity->next = node->head;
				node_entity->entity = entity_idx;
				node->head = node_entity_idx;

				++node->count;
			}
			while(node_info != node_infos);

			++insertion;
		}

		alloc_free(insertions, sizeof(*insertions) * ot->insertions_size);
		ot->insertions = NULL;
		ot->insertions_used = 0;
		ot->insertions_size = 0;
	}


	{
		uint32_t free_node = 0; /* Not reflected in ot->free_node */
		uint32_t nodes_used = ot->nodes_used;
		uint32_t nodes_size = ot->nodes_size;

		octree_node_t* new_nodes;
		octree_node_entity_t* new_node_entities;
		octree_entity_t* new_entities;

		uint32_t new_nodes_used = 0;
		uint32_t new_nodes_size;

		if(nodes_size >> 2 < nodes_used)
		{
			new_nodes_size = nodes_size;
		}
		else
		{
			new_nodes_size = nodes_size >> 1;
		}

		uint32_t new_node_entities_used = 1;
		uint32_t new_node_entities_size;

		if(node_entities_size >> 2 < node_entities_used)
		{
			new_node_entities_size = node_entities_size;
		}
		else
		{
			new_node_entities_size = node_entities_size >> 1;
		}

		uint32_t new_entities_used = 1;
		uint32_t new_entities_size;

		if(entities_size >> 2 < entities_used)
		{
			new_entities_size = entities_size;
		}
		else
		{
			new_entities_size = entities_size >> 1;
		}

		new_nodes = alloc_malloc(sizeof(*new_nodes) * new_nodes_size);
		assert_not_null(new_nodes);

		new_node_entities = alloc_malloc(sizeof(*new_node_entities) * new_node_entities_size);
		assert_not_null(new_node_entities);

		new_entities = alloc_malloc(sizeof(*new_entities) * new_entities_size);
		assert_not_null(new_entities);

		uint32_t* entity_map = alloc_calloc(sizeof(*entity_map) * entities_size);
		assert_not_null(entity_map);


		typedef struct octree_node_reorder_info
		{
			uint32_t node_idx;
			half_extent_3d_t extent;
			uint32_t parent_node_idx;
			uint32_t head_idx;
		}
		octree_node_reorder_info_t;

		octree_node_reorder_info_t node_infos[OCTREE_DFS_LENGTH];
		octree_node_reorder_info_t* node_info = node_infos;

		*(node_info++) =
		(octree_node_reorder_info_t)
		{
			.node_idx = 0,
			.extent = ot->half_extent,
			.parent_node_idx = 0,
			.head_idx = 0
		};

		do
		{
			octree_node_reorder_info_t info = *(--node_info);
			octree_node_t* node = nodes + info.node_idx;

			uint32_t new_node_idx = new_nodes_used++;
			octree_node_t* new_node = new_nodes + new_node_idx;

			new_nodes[info.parent_node_idx].heads[info.head_idx] = new_node_idx;

			if(node->count == -1)
			{
				uint32_t total = 0;
				uint32_t max_idx = 0;
				uint32_t max_count = 0;
				bool possible = true;

				for(uint32_t i = 0; i < 8; ++i)
				{
					uint32_t node_idx = node->heads[i];
					octree_node_t* node = nodes + node_idx;

					if(node->count == -1)
					{
						possible = false;
						break;
					}

					if(node->count > max_count)
					{
						max_count = node->count;
						max_idx = i;
					}

					total += node->count;
				}

				if(possible && total <= OCTREE_MERGE_THRESHOLD)
				{
					uint32_t heads[8];
					memcpy(heads, node->heads, sizeof(heads));

					octree_node_t* children[8];
					for(uint32_t i = 0; i < 8; ++i)
					{
						children[i] = nodes + heads[i];
					}

					octree_node_t* max_child = children[max_idx];
					node->count = max_child->count;
					node->head = max_child->head;
					node->position_flags = max_child->position_flags;

					for(uint32_t i = 0; i < 8; ++i)
					{
						uint32_t child_idx = heads[i];
						octree_node_t* child = children[i];

						if(i == max_idx)
						{
							child->next = free_node;
							free_node = child_idx;

							continue;
						}

						node->position_flags |= child->position_flags;

						if(!node->count)
						{
							node->count = child->count;
							node->head = child->head;

							child->next = free_node;
							free_node = child_idx;

							continue;
						}

						uint32_t node_entity_idx = child->head;
						while(node_entity_idx)
						{
							octree_node_entity_t* node_entity = node_entities + node_entity_idx;

							octree_node_entity_t* other_node_entity = node_entities + node->head;
							while(1)
							{
								if(node_entity->entity == other_node_entity->entity)
								{
									uint32_t next_node_entity_idx = node_entity->next;

									node_entity->next = free_node_entity;
									free_node_entity = node_entity_idx;

									node_entity_idx = next_node_entity_idx;

									break;
								}

								if(!other_node_entity->next)
								{
									other_node_entity->next = node_entity_idx;
									node_entity_idx = node_entity->next;
									node_entity->next = 0;

									++node->count;

									break;
								}

								other_node_entity = node_entities + other_node_entity->next;
							}
						}

						child->next = free_node;
						free_node = child_idx;
					}
				}
			}
			else if(
				node->count >= OCTREE_SPLIT_THRESHOLD &&
				info.extent.w >= ot->min_size &&
				info.extent.h >= ot->min_size &&
				info.extent.d >= ot->min_size
				)
			{
				uint32_t child_idxs[8];
				for(uint32_t i = 0; i < 8; ++i)
				{
					uint32_t child_idx;

					if(free_node)
					{
						child_idx = free_node;
						free_node = nodes[child_idx].next;
					}
					else
					{
						if(nodes_used >= nodes_size)
						{
							uint32_t new_size = (nodes_used << 1) | 1;

							nodes = alloc_remalloc(nodes,
								sizeof(*nodes) * nodes_size,
								sizeof(*nodes) * new_size);
							assert_not_null(nodes);

							nodes_size = new_size;

							node = nodes + info.node_idx;


							if(new_size > new_nodes_size)
							{
								new_nodes = alloc_remalloc(new_nodes,
									sizeof(*nodes) * new_nodes_size,
									sizeof(*nodes) * new_size);
								assert_not_null(new_nodes);

								new_nodes_size = new_size;

								new_node = new_nodes + new_node_idx;
							}
						}

						child_idx = nodes_used++;
					}

					child_idxs[i] = child_idx;
				}

				octree_node_t* children[8];
				uint32_t head = node->head;
				uint32_t position_flags = node->position_flags;

				for(uint32_t i = 0; i < 8; ++i)
				{
					uint32_t child_idx = child_idxs[i];
					octree_node_t* child = nodes + child_idx;
					children[i] = child;

					node->heads[i] = child_idx;

					child->head = 0;
					child->count = 0;

					static const uint32_t position_flags_mask[8] =
					{
						0b101010,
						0b101001,
						0b100110,
						0b100101,
						0b011010,
						0b011001,
						0b010110,
						0b010101,
					};

					child->position_flags = position_flags & position_flags_mask[i];
				}

				uint32_t node_entity_idx = head;
				while(node_entity_idx)
				{
					octree_node_entity_t* node_entity = node_entities + node_entity_idx;

					uint32_t entity_idx = node_entity->entity;
					octree_entity_t* entity = entities + entity_idx;

					rect_extent_3d_t entity_extent = octree_get_entity_rect_extent(entity);

					uint32_t target_node_idxs[8];
					uint32_t* current_target_node_idx = target_node_idxs;

					if(entity_extent.min_x <= info.extent.x)
					{
						if(entity_extent.min_y <= info.extent.y)
						{
							if(entity_extent.min_z <= info.extent.z)
							{
								*(current_target_node_idx++) = 0;
							}
							if(entity_extent.max_z >= info.extent.z)
							{
								*(current_target_node_idx++) = 1;
							}
						}
						if(entity_extent.max_y >= info.extent.y)
						{
							if(entity_extent.min_z <= info.extent.z)
							{
								*(current_target_node_idx++) = 2;
							}
							if(entity_extent.max_z >= info.extent.z)
							{
								*(current_target_node_idx++) = 3;
							}
						}
					}
					if(entity_extent.max_x >= info.extent.x)
					{
						if(entity_extent.min_y <= info.extent.y)
						{
							if(entity_extent.min_z <= info.extent.z)
							{
								*(current_target_node_idx++) = 4;
							}
							if(entity_extent.max_z >= info.extent.z)
							{
								*(current_target_node_idx++) = 5;
							}
						}
						if(entity_extent.max_y >= info.extent.y)
						{
							if(entity_extent.min_z <= info.extent.z)
							{
								*(current_target_node_idx++) = 6;
							}
							if(entity_extent.max_z >= info.extent.z)
							{
								*(current_target_node_idx++) = 7;
							}
						}
					}

					for(uint32_t* target_node_idx = target_node_idxs; target_node_idx != current_target_node_idx; ++target_node_idx)
					{
						octree_node_t* target_node = children[*target_node_idx];

						uint32_t new_node_entity_idx;
						octree_node_entity_t* new_node_entity;

						if(free_node_entity)
						{
							new_node_entity_idx = free_node_entity;
							new_node_entity = node_entities + new_node_entity_idx;
							free_node_entity = new_node_entity->next;
						}
						else
						{
							if(node_entities_used >= node_entities_size)
							{
								uint32_t new_size = (node_entities_used << 1) | 1;

								node_entities = alloc_remalloc(node_entities,
									sizeof(*node_entities) * node_entities_size,
									sizeof(*node_entities) * new_size);
								assert_not_null(node_entities);

								node_entities_size = new_size;

								node_entity = node_entities + node_entity_idx;


								if(new_size > new_node_entities_size)
								{
									new_node_entities = alloc_remalloc(new_node_entities,
										sizeof(*new_node_entities) * new_node_entities_size,
										sizeof(*new_node_entities) * new_size);
									assert_not_null(new_node_entities);

									new_node_entities_size = new_size;
								}
							}

							new_node_entity_idx = node_entities_used++;
							new_node_entity = node_entities + new_node_entity_idx;
						}

						new_node_entity->next = target_node->head;
						new_node_entity->entity = entity_idx;
						target_node->head = new_node_entity_idx;

						++target_node->count;
					}

					uint32_t next_node_entity_idx = node_entity->next;

					node_entity->next = free_node_entity;
					free_node_entity = node_entity_idx;

					node_entity_idx = next_node_entity_idx;
				}

				node->count = -1;
			}

			if(node->count == -1)
			{
				float half_w = info.extent.w * 0.5f;
				float half_h = info.extent.h * 0.5f;
				float half_d = info.extent.d * 0.5f;

				*(node_info++) =
				(octree_node_reorder_info_t)
				{
					.node_idx = node->heads[0],
					.extent =
					(half_extent_3d_t)
					{
						.x = info.extent.x - half_w,
						.y = info.extent.y - half_h,
						.z = info.extent.z - half_d,
						.w = half_w,
						.h = half_h,
						.d = half_d
					},
					.parent_node_idx = new_node_idx,
					.head_idx = 0
				};

				*(node_info++) =
				(octree_node_reorder_info_t)
				{
					.node_idx = node->heads[1],
					.extent =
					(half_extent_3d_t)
					{
						.x = info.extent.x - half_w,
						.y = info.extent.y - half_h,
						.z = info.extent.z + half_d,
						.w = half_w,
						.h = half_h,
						.d = half_d
					},
					.parent_node_idx = new_node_idx,
					.head_idx = 1
				};

				*(node_info++) =
				(octree_node_reorder_info_t)
				{
					.node_idx = node->heads[2],
					.extent =
					(half_extent_3d_t)
					{
						.x = info.extent.x - half_w,
						.y = info.extent.y + half_h,
						.z = info.extent.z - half_d,
						.w = half_w,
						.h = half_h,
						.d = half_d
					},
					.parent_node_idx = new_node_idx,
					.head_idx = 2
				};

				*(node_info++) =
				(octree_node_reorder_info_t)
				{
					.node_idx = node->heads[3],
					.extent =
					(half_extent_3d_t)
					{
						.x = info.extent.x - half_w,
						.y = info.extent.y + half_h,
						.z = info.extent.z + half_d,
						.w = half_w,
						.h = half_h,
						.d = half_d
					},
					.parent_node_idx = new_node_idx,
					.head_idx = 3
				};

				*(node_info++) =
				(octree_node_reorder_info_t)
				{
					.node_idx = node->heads[4],
					.extent =
					(half_extent_3d_t)
					{
						.x = info.extent.x + half_w,
						.y = info.extent.y - half_h,
						.z = info.extent.z - half_d,
						.w = half_w,
						.h = half_h,
						.d = half_d
					},
					.parent_node_idx = new_node_idx,
					.head_idx = 4
				};

				*(node_info++) =
				(octree_node_reorder_info_t)
				{
					.node_idx = node->heads[5],
					.extent =
					(half_extent_3d_t)
					{
						.x = info.extent.x + half_w,
						.y = info.extent.y - half_h,
						.z = info.extent.z + half_d,
						.w = half_w,
						.h = half_h,
						.d = half_d
					},
					.parent_node_idx = new_node_idx,
					.head_idx = 5
				};

				*(node_info++) =
				(octree_node_reorder_info_t)
				{
					.node_idx = node->heads[6],
					.extent =
					(half_extent_3d_t)
					{
						.x = info.extent.x + half_w,
						.y = info.extent.y + half_h,
						.z = info.extent.z - half_d,
						.w = half_w,
						.h = half_h,
						.d = half_d
					},
					.parent_node_idx = new_node_idx,
					.head_idx = 6
				};

				*(node_info++) =
				(octree_node_reorder_info_t)
				{
					.node_idx = node->heads[7],
					.extent =
					(half_extent_3d_t)
					{
						.x = info.extent.x + half_w,
						.y = info.extent.y + half_h,
						.z = info.extent.z + half_d,
						.w = half_w,
						.h = half_h,
						.d = half_d
					},
					.parent_node_idx = new_node_idx,
					.head_idx = 7
				};

				new_node->count = -1;
			}
			else
			{
				new_node->position_flags = node->position_flags;

				if(!node->head)
				{
					new_node->head = 0;
					new_node->count = 0;

					continue;
				}

				uint32_t node_entity_idx = node->head;

				new_node->head = new_node_entities_used;
				new_node->count = node->count;

				while(1)
				{
					octree_node_entity_t* node_entity = node_entities + node_entity_idx;
					octree_node_entity_t* new_node_entity = new_node_entities + new_node_entities_used;
					++new_node_entities_used;

					uint32_t entity_idx = node_entity->entity;
					if(!entity_map[entity_idx])
					{
						uint32_t new_entity_idx = new_entities_used++;
						entity_map[entity_idx] = new_entity_idx;
						new_entities[new_entity_idx] = entities[entity_idx];
					}

					new_node_entity->entity = entity_map[entity_idx];

					if(node_entity->next)
					{
						node_entity_idx = node_entity->next;
						new_node_entity->next = new_node_entities_used;
					}
					else
					{
						new_node_entity->next = 0;
						break;
					}
				}
			}
		}
		while(node_info != node_infos);

		alloc_free(nodes, sizeof(*nodes) * nodes_size);
		ot->nodes = new_nodes;
		ot->nodes_used = new_nodes_used;
		ot->nodes_size = new_nodes_size;

		alloc_free(node_entities, sizeof(*node_entities) * node_entities_size);
		ot->node_entities = new_node_entities;
		ot->node_entities_used = new_node_entities_used;
		ot->node_entities_size = new_node_entities_size;

		alloc_free(entities, sizeof(*entities) * entities_size);
		ot->entities = new_entities;
		ot->entities_used = new_entities_used;
		ot->entities_size = new_entities_size;

		alloc_free(entity_map, sizeof(*entity_map) * entities_size);
	}
}


void
octree_update(
	octree_t* ot,
	octree_update_fn_t update_fn
	)
{
	ot->update_tick ^= 1;
	uint32_t update_tick = ot->update_tick;

	octree_node_t* nodes = ot->nodes;
	octree_node_entity_t* node_entities = ot->node_entities;
	octree_entity_t* entities = ot->entities;
	octree_reinsertion_t* reinsertions = ot->reinsertions;
	octree_node_removal_t* node_removals = ot->node_removals;

	uint32_t reinsertions_used = ot->reinsertions_used;
	uint32_t reinsertions_size = ot->reinsertions_size;

	uint32_t node_removals_used = ot->node_removals_used;
	uint32_t node_removals_size = ot->node_removals_size;

	octree_node_info_t node_infos[OCTREE_DFS_LENGTH];
	octree_node_info_t* node_info = node_infos;

	*(node_info++) =
	(octree_node_info_t)
	{
		.node_idx = 0,
		.extent = ot->half_extent
	};

	do
	{
		octree_node_info_t info = *(--node_info);
		octree_node_t* node = nodes + info.node_idx;

		if(node->count == -1)
		{
			octree_descend((
				(rect_extent_3d_t)
				{
					.min = {{ info.extent.x, info.extent.y, info.extent.z }},
					.max = {{ info.extent.x, info.extent.y, info.extent.z }}
				}
				));
			continue;
		}

		rect_extent_3d_t node_extent = half_to_rect_3d_extent(info.extent);

		uint32_t prev_idx = 0;
		uint32_t idx = node->head;

		while(idx)
		{
			octree_node_entity_t* node_entity = node_entities + idx;

			uint32_t entity_idx = node_entity->entity;
			octree_entity_t* entity = entities + entity_idx;

			rect_extent_3d_t extent;

			if(entity->update_tick != update_tick)
			{
				entity->update_tick = update_tick;
				octree_status_t status = update_fn(ot, entity_idx, &entity->data);
				extent = octree_get_entity_rect_extent(entity);

				if(status == OCTREE_STATUS_CHANGED)
				{
					entity->fully_in_node = rect_extent_3d_is_inside(extent, node_extent);
					if(!entity->fully_in_node)
					{
						uint32_t reinsertion_idx;
						octree_reinsertion_t* reinsertion;

						if(reinsertions_used >= reinsertions_size)
						{
							uint32_t new_size = (reinsertions_used << 1) | 1;

							reinsertions = alloc_remalloc(reinsertions,
								sizeof(*reinsertions) * reinsertions_size,
								sizeof(*reinsertions) * new_size);
							assert_not_null(reinsertions);

							reinsertions_size = new_size;
						}

						reinsertion_idx = reinsertions_used++;
						reinsertion = reinsertions + reinsertion_idx;

						reinsertion->entity_idx = entity_idx;
					}
				}
			}
			else
			{
				extent = octree_get_entity_rect_extent(entity);
			}

			if(
				(extent.max_x < node_extent.min_x && !(node->position_flags & 0b100000)) ||
				(extent.max_y < node_extent.min_y && !(node->position_flags & 0b001000)) ||
				(extent.max_z < node_extent.min_z && !(node->position_flags & 0b000010)) ||
				(node_extent.max_x < extent.min_x && !(node->position_flags & 0b010000)) ||
				(node_extent.max_y < extent.min_y && !(node->position_flags & 0b000100)) ||
				(node_extent.max_z < extent.min_z && !(node->position_flags & 0b000001))
				)
			{
				uint32_t node_removal_idx;
				octree_node_removal_t* node_removal;

				if(node_removals_used >= node_removals_size)
				{
					uint32_t new_size = (node_removals_used << 1) | 1;

					node_removals = alloc_remalloc(node_removals,
						sizeof(*node_removals) * node_removals_size,
						sizeof(*node_removals) * new_size);
					assert_not_null(node_removals);

					node_removals_size = new_size;
				}

				node_removal_idx = node_removals_used++;
				node_removal = node_removals + node_removal_idx;

				node_removal->node_idx = info.node_idx;
				node_removal->node_entity_idx = idx;
				node_removal->prev_node_entity_idx = prev_idx;
			}

			prev_idx = idx;
			idx = node_entity->next;
		}
	}
	while(node_info != node_infos);

	ot->reinsertions = reinsertions;
	ot->reinsertions_used = reinsertions_used;
	ot->reinsertions_size = reinsertions_size;

	ot->node_removals = node_removals;
	ot->node_removals_used = node_removals_used;
	ot->node_removals_size = node_removals_size;
}


void
octree_query(
	octree_t* ot,
	rect_extent_3d_t extent,
	octree_query_fn_t query_fn
	)
{
	++ot->query_tick;
	uint32_t query_tick = ot->query_tick;

	octree_node_t* nodes = ot->nodes;
	octree_node_entity_t* node_entities = ot->node_entities;
	octree_entity_t* entities = ot->entities;

	octree_node_info_t node_infos[OCTREE_DFS_LENGTH];
	octree_node_info_t* node_info = node_infos;

	*(node_info++) =
	(octree_node_info_t)
	{
		.node_idx = 0,
		.extent = ot->half_extent
	};

	do
	{
		octree_node_info_t info = *(--node_info);
		octree_node_t* node = nodes + info.node_idx;

		if(node->count == -1)
		{
			octree_descend(extent);
			continue;
		}

		uint32_t idx = node->head;
		while(idx)
		{
			octree_node_entity_t* node_entity = node_entities + idx;
			uint32_t entity_idx = node_entity->entity;
			octree_entity_t* entity = entities + entity_idx;

			if(entity->query_tick != query_tick)
			{
				entity->query_tick = query_tick;

				if(rect_extent_3d_intersects(octree_get_entity_rect_extent(entity), extent))
				{
					query_fn(ot, entity_idx, &entity->data);
				}
			}

			idx = node_entity->next;
		}
	}
	while(node_info != node_infos);
}


void
octree_query_nodes(
	octree_t* ot,
	rect_extent_3d_t extent,
	octree_node_query_fn_t node_query_fn
	)
{
	octree_node_t* nodes = ot->nodes;

	octree_node_info_t node_infos[OCTREE_DFS_LENGTH];
	octree_node_info_t* node_info = node_infos;

	*(node_info++) =
	(octree_node_info_t)
	{
		.node_idx = 0,
		.extent = ot->half_extent
	};

	do
	{
		octree_node_info_t info = *(--node_info);
		octree_node_t* node = nodes + info.node_idx;

		if(node->count == -1)
		{
			octree_descend(extent);
			continue;
		}

		node_query_fn(ot, &info);
	}
	while(node_info != node_infos);
}


void
octree_collide(
	octree_t* ot,
	octree_collide_fn_t collide_fn
	)
{
	octree_normalize(ot);

	if(ot->entities_used <= 1)
	{
		return;
	}

#if OCTREE_DEDUPE_COLLISIONS == 1
	uint32_t ht_size = ot->ht_entries_used * 2;
	uint32_t* ht = alloc_calloc(sizeof(*ht) * ht_size);
	assert_not_null(ht);

	octree_ht_entry_t* ht_entries = ot->ht_entries;

	uint32_t ht_entries_used = 1;
	uint32_t ht_entries_size = ot->ht_entries_size;
#endif

	octree_node_entity_t* node_entities = ot->node_entities;
	octree_entity_t* entities = ot->entities;

	octree_node_entity_t* node_entity = node_entities;
	octree_node_entity_t* node_entities_end = node_entities + ot->node_entities_used - 1;

	do
	{
		++node_entity;

		if(!node_entity->next)
		{
			continue;
		}

		uint32_t entity_idx = node_entity->entity;
		octree_entity_t* entity = entities + entity_idx;
		rect_extent_3d_t entity_extent = octree_get_entity_rect_extent(entity);

		octree_node_entity_t* other_node_entity = node_entity;
		while(1)
		{
			++other_node_entity;

			uint32_t other_entity_idx = other_node_entity->entity;
			octree_entity_t* other_entity = entities + other_entity_idx;

			if(!rect_extent_3d_intersects(
				entity_extent,
				octree_get_entity_rect_extent(other_entity)
				))
			{
				goto goto_skip;
			}

#if OCTREE_DEDUPE_COLLISIONS == 1
			if(!entity->fully_in_node || !other_entity->fully_in_node)
			{
				uint32_t index_a = entity_idx;
				uint32_t index_b = other_entity_idx;

				if(index_a > index_b)
				{
					uint32_t temp = index_a;
					index_a = index_b;
					index_b = temp;
				}

				uint32_t hash = index_a * 48611 + index_b * 50261;
				hash %= ht_size;

				uint32_t index = ht[hash];
				octree_ht_entry_t* entry;

				while(index)
				{
					entry = ht_entries + index;

					if(entry->idx[0] == index_a && entry->idx[1] == index_b)
					{
						goto goto_skip;
					}

					index = entry->next;
				}

				if(ht_entries_used >= ht_entries_size)
				{
					uint32_t new_size = (ht_entries_used << 1) | 1;

					ht_entries = alloc_remalloc(ht_entries,
						sizeof(*ht_entries) * ht_entries_size,
						sizeof(*ht_entries) * new_size);
					assert_not_null(ht_entries);

					ht_entries_size = new_size;
				}

				uint32_t entry_idx = ht_entries_used++;
				entry = ht_entries + entry_idx;

				entry->idx[0] = index_a;
				entry->idx[1] = index_b;
				entry->next = index;
				ht[hash] = entry_idx;
			}
#endif

			collide_fn(ot, &entity->data, &other_entity->data);

			goto_skip:;

			if(!other_node_entity->next)
			{
				break;
			}
		}
	}
	while(node_entity != node_entities_end);

#if OCTREE_DEDUPE_COLLISIONS == 1
	if(ht_entries_used * 4 <= ht_entries_size)
	{
		uint32_t new_size = ht_entries_size >> 1;
		ht_entries = alloc_remalloc(ht_entries,
			sizeof(*ht_entries) * ht_entries_size,
			sizeof(*ht_entries) * new_size);
		assert_not_null(ht_entries);

		ht_entries_size = new_size;
	}

	ot->ht_entries = ht_entries;
	ot->ht_entries_used = ht_entries_used;
	ot->ht_entries_size = ht_entries_size;

	alloc_free(ht, sizeof(*ht) * ht_size);
#endif
}


#undef octree_descend
